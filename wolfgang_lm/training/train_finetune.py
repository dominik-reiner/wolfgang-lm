from typing import Literal
import os
import csv
import time
import math
import torch
from dataclasses import asdict
from wolfgang_lm.modeling.config import ModelConfig
from wolfgang_lm.training.config import TrainingConfig
from wolfgang_lm.modeling.transformer import WolfgangGPT


def train(
    train_config: TrainingConfig, model_config: ModelConfig, pretrained_path: str
):
    # -----------------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------------
    print("Fine-Tuning Configuration:")
    print(train_config)
    print("\nModel Configuration:")
    print(model_config)

    os.makedirs(train_config.out_dir, exist_ok=True)

    # Initialize log files
    train_log_path = os.path.join(train_config.out_dir, "train_finetune_log.csv")
    val_log_path = os.path.join(train_config.out_dir, "val_finetune_log.csv")

    # Write headers
    with open(train_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "loss", "ppl", "grad_norm", "lr", "time_ms"])

    with open(val_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "train_loss", "train_ppl", "val_loss", "val_ppl"])
    torch.manual_seed(1337)

    # Enable tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if "cuda" in train_config.device and not torch.cuda.is_available():
        print("Warning: CUDA not available. Switching to MPS/CPU.")
        if torch.backends.mps.is_available():
            train_config.device = "mps"
        else:
            train_config.device = "cpu"
        train_config.compile = False

    device_type = "cuda" if "cuda" in train_config.device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[train_config.dtype]

    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # -----------------------------------------------------------------------------
    # Data Loader (Sample-Aligned for Fine-Tuning)
    # -----------------------------------------------------------------------------
    data_path = os.path.join(train_config.dataset, "finetune_dataset.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run prepare_finetune.py."
        )

    print(f"Loading dataset from {data_path}...")
    dataset = torch.load(data_path)
    train_samples = dataset["train"]
    val_samples = dataset["val"]

    # Load Config from dataset
    ds_config = dataset.get("config", {})
    pad_id = ds_config.get(
        "pad_id", 0
    )  # Fallback to 0 if not found (but should be there)
    print(
        f"Loaded {len(train_samples)} training samples and {len(val_samples)} validation samples."
    )
    print(f"Using Padding ID: {pad_id}")

    def get_batch(split: Literal["train", "val"]):
        samples = train_samples if split == "train" else val_samples

        # 1. Sample Random Indices
        # We pick random whole conversations
        batch_indices = torch.randint(0, len(samples), (train_config.batch_size,))

        # 2. Batch Creation with Padding
        # Initialize x with padding (pad_id) and y with ignore_index (-1)

        x = torch.full(
            (train_config.batch_size, model_config.block_size), pad_id, dtype=torch.long
        )
        y = torch.full(
            (train_config.batch_size, model_config.block_size), -1, dtype=torch.long
        )

        for k, idx in enumerate(batch_indices):
            s = samples[idx.item()]
            inp = s["input_ids"]
            tgt = s["labels"]

            # Truncate if necessary (though prep script should handle this)
            length = min(len(inp), model_config.block_size)

            # Safe truncation length for x/y pair involves L-1 tokens
            use_len = length - 1
            if use_len < 1:
                continue  # Skip empty/single-token garbage

            x[k, :use_len] = torch.tensor(inp[:use_len], dtype=torch.long)
            y[k, :use_len] = torch.tensor(tgt[1:length], dtype=torch.long)

        if device_type == "cuda":
            x = x.pin_memory().to(train_config.device, non_blocking=True)
            y = y.pin_memory().to(train_config.device, non_blocking=True)
        else:
            x = x.to(train_config.device)
            y = y.to(train_config.device)

        return x, y

    # -----------------------------------------------------------------------------
    # Model Init & Loading
    # -----------------------------------------------------------------------------
    print("Initializing Model...")
    model = WolfgangGPT(model_config)

    # LOAD PRETRAINED WEIGHTS
    print(f"Loading pretrained weights from {pretrained_path}...")
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=train_config.device)
        # Handle both raw state_dict or full checkpoint dict
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

        # In case key names have 'module.' prefix (from DDP)
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    else:
        print(f"Error: Pretrained checkpoint {pretrained_path} not found.")
        # For development/testing, we might want to continue or exit.
        # Here we'll raise error as fine-tuning requires a base model.
        raise FileNotFoundError(f"Pretrained checkpoint {pretrained_path} not found.")

    model.to(train_config.device)

    print(f"Compiling Model: {train_config.compile}")
    if train_config.compile:
        model = torch.compile(model)

    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=train_config.learning_rate,
        betas=(0.9, 0.95),
        device_type=device_type,
    )

    # -----------------------------------------------------------------------------
    # Scheduling & Training Loop (Simplified for Finetune)
    # -----------------------------------------------------------------------------

    def get_lr(it):
        # Linear Warmup
        if it < train_config.warmup_iters:
            return train_config.learning_rate * it / train_config.warmup_iters
        # Cosine Decay
        if it > train_config.lr_decay_iters:
            return train_config.min_lr
        decay_ratio = (it - train_config.warmup_iters) / (
            train_config.lr_decay_iters - train_config.warmup_iters
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return train_config.min_lr + coeff * (
            train_config.learning_rate - train_config.min_lr
        )

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(train_config.eval_iters)
            for k in range(train_config.eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    X, Y = get_batch("train")
    t0 = time.time()
    best_val_loss = 1e9

    print("Starting Fine-Tuning...")

    for iter_num in range(train_config.max_iters):
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward / Backward
        for micro_step in range(train_config.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / train_config.gradient_accumulation_steps
            loss.backward()
            X, Y = get_batch("train")

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Eval
        if iter_num % train_config.eval_interval == 0:
            losses = estimate_loss()
            print(
                "EVAL",
                f"step {iter_num}: "
                f"train loss {losses['train']:.4f}, "
                f"train ppl {math.exp(losses['train']):.4f}, "
                f"val loss {losses['val']:.4f}, "
                f"val ppl {math.exp(losses['val']):.4f}",
            )
            with open(val_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        iter_num,
                        losses["train"].item(),
                        math.exp(losses["train"]),
                        losses["val"].item(),
                        math.exp(losses["val"]),
                    ]
                )

            if losses["val"] < best_val_loss or train_config.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": asdict(model_config),
                        "train_config": asdict(train_config),
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                    }
                    path = os.path.join(train_config.out_dir, "finetune_ckpt.pt")
                    print(f"Saving checkpoint to {path}")
                    torch.save(checkpoint, path)

        # Log
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % train_config.log_interval == 0:
            lossf = loss.item() * train_config.gradient_accumulation_steps
            try:
                ppl = math.exp(lossf)
            except OverflowError:
                ppl = float("inf")

            print(
                f"iter {iter_num}: loss {lossf:.4f}, ppl {ppl:.2f}, "
                f"norm {grad_norm:.4f}, time {dt*1000:.2f}ms, lr {lr:.4e}"
            )
            with open(train_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([iter_num, lossf, ppl, grad_norm.item(), lr, dt * 1000])

    # -----------------------------------------------------------------------------
    # Save Final Checkpoint
    # -----------------------------------------------------------------------------
    print("Saving Final Fine-Tuning Checkpoint...")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "iter_num": train_config.max_iters,
        "best_val_loss": best_val_loss,
    }
    path = os.path.join(train_config.out_dir, "finetune_ckpt_final.pt")
    print(f"Saving checkpoint to {path}")
    torch.save(checkpoint, path)

    print("Fine-Tuning Complete.")
    return best_val_loss


if __name__ == "__main__":
    # Config Overrides for Fine-Tuning
    train_config = TrainingConfig()
    train_config.out_dir = "out-finetune"

    # Reduce LR for fine-tuning
    train_config.learning_rate = (
        5e-5  # Lower than pretrain min_lr (6e-5) to avoid shock
    )
    train_config.min_lr = 5e-6

    # Preventing Overfitting:
    # Dataset is ~4500 samples.
    # Batch Size 8 * Grad Accum 4 = 32 samples/step.
    # ~140 steps per epoch (4500 / 32).
    # Target: ~5.0 epochs -> ~700 steps.
    train_config.batch_size = 8
    train_config.gradient_accumulation_steps = 4
    train_config.max_iters = 700
    train_config.warmup_iters = 70  # Warmup for first ~10%
    train_config.lr_decay_iters = 700  # Decay down to min_lr by end
    train_config.eval_interval = 50
    train_config.log_interval = 10

    model_config = ModelConfig()

    pretrained_ckpt = "out-pretrain/ckpt_final.pt"
    if not os.path.exists(pretrained_ckpt):
        # Fallback to just ckpt.pt if final not there (common in early stop)
        if os.path.exists("out-pretrain/ckpt.pt"):
            pretrained_ckpt = "out-pretrain/ckpt.pt"

    train(train_config, model_config, pretrained_ckpt)

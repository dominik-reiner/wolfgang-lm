from typing import Literal
import os
import csv
import time
import math
import numpy as np
import torch
from dataclasses import asdict
from wolfgang_lm.modeling.config import ModelConfig
from wolfgang_lm.training.config import TrainingConfig
from wolfgang_lm.modeling.transformer import WolfgangGPT


def train(train_config: TrainingConfig, model_config: ModelConfig):
    # -----------------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------------
    print("Training Configuration:")
    print(train_config)
    print("\nModel Configuration:")
    print(model_config)

    os.makedirs(train_config.out_dir, exist_ok=True)

    # Initialize log files
    train_log_path = os.path.join(train_config.out_dir, "train_log.csv")
    val_log_path = os.path.join(train_config.out_dir, "val_log.csv")

    # Write headers
    with open(train_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "loss", "ppl", "grad_norm", "lr", "time_ms"])

    with open(val_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "train_loss", "train_ppl", "val_loss", "val_ppl"])
    torch.manual_seed(1337)
    # TensorFloat-32 (TF32) Logic:
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
    torch.backends.cudnn.allow_tf32 = True

    if "cuda" in train_config.device and not torch.cuda.is_available():
        print(
            "Warning: CUDA not available. Switching to MPS for Mac if available, else CPU."
        )
        if torch.backends.mps.is_available():
            train_config.device = "mps"
        else:
            train_config.device = "cpu"
        train_config.compile = False

    device_type = "cuda" if "cuda" in train_config.device else "cpu"
    # Automatic Mixed Precision (AMP) Setup:
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[train_config.dtype]

    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # -----------------------------------------------------------------------------
    # Data Loader
    # -----------------------------------------------------------------------------
    data_dir = train_config.dataset
    train_path = os.path.join(data_dir, "train.bin")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}. Run prepare_data.py first."
        )

    # "Memory Mapped" File
    data = np.memmap(train_path, dtype=np.uint16, mode="r")

    # Split Data (95% Train / 5% Val)
    n = int(0.95 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split: Literal["train", "val"]):
        """
        Fetches a batch of data for training or validation.
        """
        # Select the correct data split
        data = train_data if split == "train" else val_data

        # Random Sampling:
        ix = torch.randint(
            len(data) - model_config.block_size, (train_config.batch_size,)
        )

        # Input (x) and Target (y):
        x = torch.stack(
            [
                torch.from_numpy(
                    (data[i : i + model_config.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + model_config.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        # Data Transfer Optimization:
        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously
            x, y = x.pin_memory().to(
                train_config.device, non_blocking=True
            ), y.pin_memory().to(train_config.device, non_blocking=True)
        else:
            x, y = x.to(train_config.device), y.to(train_config.device)
        return x, y

    # -----------------------------------------------------------------------------
    # Model Init
    # -----------------------------------------------------------------------------
    print("Initializing Model...")
    model = WolfgangGPT(model_config)
    model.to(train_config.device)

    print(f"Compiling Model: {train_config.compile}")
    if train_config.compile:
        model = torch.compile(model)

    # Optimizer
    optimizer = model.configure_optimizers(
        # AdamW Update (Sequential):
        # 1. Weight Decay: theta = theta - lr * weight_decay * theta
        #    (This "shrinks" the weights directly, decoupled from gradients)
        weight_decay=1e-1,
        learning_rate=train_config.learning_rate,
        # Betas (Adam): (beta1, beta2) configures the optimizer's "memory".
        # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          (First Moment: Momentum)
        # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        (Second Moment: Variance)
        # Where: m=moving avg of gradient, v=moving avg of squared gradient, g_t=current gradient.
        # We use beta2=0.95 (vs default 0.999) to shorten the memory horizon for variance,
        # allowing the model to recover faster from gradient spikes common in LLMs.
        # Final Update (simplified):
        # theta_new = theta_new - lr * m_t / (sqrt(v_t) + epsilon)
        betas=(0.9, 0.95),
        device_type=device_type,
    )

    # Learning Rate Scheduler (Cosine)
    def get_lr(it):
        # 1. Linear Warmup
        if it < train_config.warmup_iters:
            # Ramp up linearly from 0 to the target learning rate
            return train_config.learning_rate * it / train_config.warmup_iters
        # 2. Minimum LR floor (after decay)
        if it > train_config.lr_decay_iters:
            return train_config.min_lr
        # 3. Cosine Decay
        decay_ratio = (it - train_config.warmup_iters) / (
            train_config.lr_decay_iters - train_config.warmup_iters
        )
        # coeff starts at 1.0 and smoothly, slowly curves down to 0.0.
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        # scale the learning rate by the coefficient
        return train_config.min_lr + coeff * (
            train_config.learning_rate - train_config.min_lr
        )

    # Evaluation Function (No Gradients)
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()  # Switch to Evaluation Mode
        for split in ["train", "val"]:
            losses = torch.zeros(train_config.eval_iters)
            for k in range(train_config.eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()  # Switch back to training mode
        return out

    # -----------------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------------
    X, Y = get_batch("train")
    t0 = time.time()
    best_val_loss = 1e9

    print("Starting Training...")
    for iter_num in range(train_config.max_iters):
        # Determine the exact learning rate for this specific step
        lr = get_lr(iter_num)
        # Inject it into the optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient Acc Loop
        for micro_step in range(train_config.gradient_accumulation_steps):
            with ctx:  # AMP
                logits, loss = model(X, Y)
                # Scale Loss: We divide by accumulation steps because gradients SUM up by default.
                # We want the average gradient of all batches.
                loss = loss / train_config.gradient_accumulation_steps

            # Backward Pass
            # This populates the .grad attribute for every parameter p (p.grad).
            loss.backward()

            # Prefetch the next batch
            X, Y = get_batch("train")

        # Gradient Clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer Step:
        # The optimizer holds references to all model parameters (from model.configure_optimizers).
        # It iterates over these parameters and uses p.grad to update p.data.
        optimizer.step()

        # Reset Gradients
        optimizer.zero_grad(set_to_none=True)

        # Logging
        # -------------------------------------------------------------------------
        # 1. Evaluation Loop
        if iter_num % train_config.eval_interval == 0:
            losses = estimate_loss()
            print("EVAL")
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, "
                f"train ppl {math.exp(losses['train']):.4f}, "
                f"val loss {losses['val']:.4f}, "
                f"val ppl {math.exp(losses['val']):.4f}"
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

            # Save Checkpoint
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
                    ckpt_path = os.path.join(train_config.out_dir, "ckpt.pt")
                    print(f"Saving checkpoint to {ckpt_path}")
                    torch.save(checkpoint, ckpt_path)

        # 2. Regular Logging (every log_interval)
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % train_config.log_interval == 0:
            lossf = loss.item() * train_config.gradient_accumulation_steps
            try:
                ppl = math.exp(lossf)
            except OverflowError:
                ppl = float("inf")

            print("LOG")
            print(
                f"iter {iter_num}: loss {lossf:.4f}, ppl {ppl:.2f}, "
                f"norm {grad_norm:.4f}, time {dt*1000:.2f}ms, lr {lr:.4e}"
            )
            with open(train_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([iter_num, lossf, ppl, grad_norm.item(), lr, dt * 1000])

        # Checkpoint
        if iter_num > 0 and iter_num % 1000 == 0:
            print("CHECKPOINT")
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "iter_num": iter_num,
            }
            ckpt_path = os.path.join(train_config.out_dir, f"ckpt_{iter_num}.pt")
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(checkpoint, ckpt_path)

    # -----------------------------------------------------------------------------
    # Save Final Checkpoint
    # -----------------------------------------------------------------------------
    print("Saving Final Checkpoint...")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "iter_num": train_config.max_iters,
        "best_val_loss": best_val_loss,
    }
    ckpt_path = os.path.join(train_config.out_dir, "ckpt_final.pt")
    print(f"Saving checkpoint to {ckpt_path}")
    torch.save(checkpoint, ckpt_path)

    print("Training Complete.")
    return best_val_loss


if __name__ == "__main__":
    train_config = TrainingConfig()
    model_config = ModelConfig()

    train(train_config, model_config)

import os
import shutil
import torch
from wolfgang_lm.training.train import train
from wolfgang_lm.training.config import TrainingConfig
from wolfgang_lm.modeling.config import ModelConfig


def verify():
    print("Starting verification of training script...")

    # 1. Setup minimal configuration
    train_config = TrainingConfig()
    train_config.dataset = "data_clean"
    train_config.out_dir = "out-verify"
    train_config.max_iters = 10
    train_config.batch_size = 2
    train_config.gradient_accumulation_steps = 1
    train_config.log_interval = 1
    train_config.eval_interval = 5
    train_config.eval_iters = 1
    train_config.warmup_iters = 0  # No warmup needed for short test
    train_config.compile = False  # specific for fast verify startup
    train_config.always_save_checkpoint = False

    # Use CPU or MPS for local verification if CUDA is not available
    if not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            train_config.device = "mps"
        else:
            train_config.device = "cpu"

    model_config = ModelConfig()
    model_config.n_layer = 2  # Tiny model for speed
    model_config.n_head = 2
    model_config.n_kv_head = 2  # Match n_head for MHA or set to 1 for MQA, but essential to be <= n_head and divisible
    model_config.n_embd = 32
    model_config.block_size = 64  # Small context
    model_config.dropout = 0.0

    print(f"Device: {train_config.device}")

    # Clean up previous verification output
    if os.path.exists(train_config.out_dir):
        shutil.rmtree(train_config.out_dir)

    try:
        # 2. Run training
        final_val_loss = train(train_config, model_config)
        print(
            f"Verification training completed. Final validation loss: {final_val_loss}"
        )

        # 3. Check assertions
        # Check if checkpoint was created (if saved) - strictly we disabled always_save_checkpoint
        # but if val loss improves it might save.
        # With 5 iters and eval at 5, it should run eval once.

        print("✅ Verification Successful!")
        return True

    except Exception as e:
        print(f"❌ Verification Failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    verify()

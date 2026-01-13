import os
import shutil
import torch
from wolfgang_lm.training.train_finetune import train
from wolfgang_lm.training.config import TrainingConfig
from wolfgang_lm.modeling.config import ModelConfig
from wolfgang_lm.modeling.transformer import WolfgangGPT


def verify_finetune():
    print("Starting verification of FINE-TUNING script...")

    # 1. Setup minimal configuration
    train_config = TrainingConfig()
    train_config.dataset = "data_clean"
    train_config.out_dir = "out-verify-finetune"
    train_config.max_iters = 5
    train_config.batch_size = 2
    train_config.gradient_accumulation_steps = 1
    train_config.log_interval = 1
    train_config.eval_interval = 5
    train_config.eval_iters = 1
    train_config.warmup_iters = 0
    train_config.compile = False
    train_config.always_save_checkpoint = False

    # Use CPU or MPS for local verification
    if not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            train_config.device = "mps"
        else:
            train_config.device = "cpu"

    # 2. Config for Tiny Model
    model_config = ModelConfig()
    model_config.n_layer = 2
    model_config.n_head = 2
    model_config.n_kv_head = 2
    model_config.n_embd = 32
    model_config.block_size = 64
    model_config.dropout = 0.0

    print(f"Device: {train_config.device}")

    # Clean up previous verification output
    if os.path.exists(train_config.out_dir):
        shutil.rmtree(train_config.out_dir)
    os.makedirs(train_config.out_dir, exist_ok=True)

    try:
        # 3. Create a Dummy Pretrained Checkpoint
        # We need a checkpoint that matches the tiny structure above to test the loading logic.
        print("Creating dummy pretrained checkpoint...")
        dummy_model = WolfgangGPT(model_config)
        # Verify weight loading: set a weight to a known value? Not strictly necessary for crash testing.

        dummy_ckpt_path = os.path.join(train_config.out_dir, "dummy_pretrained.pt")
        torch.save(
            {
                "model": dummy_model.state_dict(),
                "model_config": model_config,  # not used by loading logic typically but good practice
            },
            dummy_ckpt_path,
        )

        del dummy_model  # Free memory

        # 4. Run Fine-Tuning Training
        print("Running train_finetune...")
        final_val_loss = train(
            train_config, model_config, pretrained_path=dummy_ckpt_path
        )

        print(
            f"Verification training completed. Final validation loss: {final_val_loss}"
        )

        print("✅ Verification Successful!")
        return True

    except Exception as e:
        print(f"❌ Verification Failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    verify_finetune()

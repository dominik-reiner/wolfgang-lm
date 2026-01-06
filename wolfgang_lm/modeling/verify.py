import torch
from .config import ModelConfig
from .transformer import WolfgangGPT


def verify_model():
    print("Verifying WolfgangGPT Architecture...")
    config = ModelConfig()
    model = WolfgangGPT(config)

    # Calculate Params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {n_params:,}")
    print("Target: ~80,000,000")

    # Detailed Breakdown
    n_embed = sum(p.numel() for p in model.embeddings.parameters())
    print(f"Embeddings: {n_embed:,}")

    # Dummy Forward Pass
    print("\nRunning Dummy Forward Pass...")
    try:
        idx = torch.randint(0, config.vocab_size, (2, 64))  # Batch=2, Time=64
        targets = torch.randint(0, config.vocab_size, (2, 64))

        logits, loss = model(idx, targets)

        print(f"Logits Shape: {logits.shape} (Expected: [2, 64, {config.vocab_size}])")
        print(f"Loss: {loss.item()}")
        print("Forward Pass Successful!")

    except Exception as e:
        print(f"Forward Pass Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    verify_model()

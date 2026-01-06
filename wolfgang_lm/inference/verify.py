import os
from wolfgang_lm.inference.generation import WolfgangGenerator


def test_manual_generation():
    # Paths (adjust if running from a different directory)
    # Assuming running from project root
    ckpt_path = "out-pretrain/ckpt_final.pt"
    tokenizer_path = "data_clean/tokenizer.json"

    # Check if files exist
    if not os.path.exists(ckpt_path):
        # Fallback to a possibly existing checkpoint for testing purposes if final doesn't exist?
        # Or just fail and let the user know.
        # Let's try 'out-pretrain/ckpt.pt' akin to the original script if final is missing
        if os.path.exists("out-verify/ckpt.pt"):
            ckpt_path = "out-verify/ckpt.pt"
            print(f"Using {ckpt_path}")
        else:
            print(f"Error: Checkpoint not found at {ckpt_path} or out-verify/ckpt.pt")
            return

    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        return

    print("Initializing Generator...")
    generator = WolfgangGenerator(ckpt_path, tokenizer_path)

    prompt = "Wer ist der faust?"
    print(f"Prompt: {prompt}")

    print("Generating (with prompt)...")
    output = generator.generate(prompt, max_new_tokens=50, include_prompt=True)

    print("\n--- Output (Full) ---")
    print(output)
    print("---------------------")

    print("Generating (without prompt)...")
    output_new = generator.generate(prompt, max_new_tokens=50, include_prompt=False)

    print("\n--- Output (New Only) ---")
    print(output_new)
    print("-------------------------")


if __name__ == "__main__":
    test_manual_generation()

import os
from wolfgang_lm.inference.generation import WolfgangGenerator


def test_manual_generation():
    # Paths (adjust if running from a different directory)
    # Assuming running from project root
    ckpt_path = "out-pretrain/ckpt_final.pt"
    tokenizer_path = "data_clean/tokenizer.json"

    # Check if files exist
    if not os.path.exists(ckpt_path):
        # Fallback
        if os.path.exists("out-pretrain/ckpt.pt"):
            ckpt_path = "out-pretrain/ckpt.pt"
            print(f"Using {ckpt_path}")
        else:
            print(f"Error: Checkpoint not found at {ckpt_path}")
            return

    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        return

    print("Initializing Generator...")
    generator = WolfgangGenerator(ckpt_path, tokenizer_path)

    prompts = [
        "Die Natur ist",
        "Was ist der Sinn des Lebens?",
        "Zwei Seelen wohnen",
        "Ach, mein Freund,",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("--- GENERATING ---")
        output = generator.generate(prompt, max_new_tokens=50, include_prompt=True)
        print(output)
        print("------------------")


if __name__ == "__main__":
    test_manual_generation()

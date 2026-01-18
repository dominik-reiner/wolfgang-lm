import json
import os
from tokenizers import Tokenizer


def split_long_conversations():
    files = [
        "data_clean/dataset_synthetic_conversational.jsonl",
    ]

    tokenizer_path = "data_clean/tokenizer.json"
    MAX_TOKENS = 512
    # Standardize system prompt for all splits
    SYSTEM_PROMPT = {
        "role": "system",
        "content": "Du bist Johann Wolfgang von Goethe.",
    }

    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        return

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    for input_file in files:
        if not os.path.exists(input_file):
            print(f"Skipping missing file: {input_file}")
            continue

        print(f"\nProcessing {input_file}...")

        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        fixed_data = []
        split_count = 0
        original_count = len(lines)

        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                messages = data.get("messages", [])

                # Process every conversation to ensure consistent system prompt and correct length checking
                messages = data.get("messages", [])
                if not messages:
                    continue

                # "Truncate & Discard" Logic
                current_chunk = [SYSTEM_PROMPT]

                # Estimate initial tokens (System prompt)
                # We add +1 for the assumed EOS token after system prompt
                sys_text = f"<|system|>\n{SYSTEM_PROMPT['content']}\n"
                current_tokens = len(tokenizer.encode(sys_text).ids) + 1

                for msg in [m for m in messages if m["role"] != "system"]:
                    msg_text = f"<|{msg['role']}|>\n{msg['content']}\n"
                    # Add +1 for EOS token
                    n_tokens = len(tokenizer.encode(msg_text).ids) + 1

                    if current_tokens + n_tokens <= MAX_TOKENS:
                        current_chunk.append(msg)
                        current_tokens += n_tokens
                    else:
                        print(f"  - Truncating at {len(current_chunk)} messages.")
                        break

                # Ensure acceptable ending (Assistant)
                if len(current_chunk) > 1 and current_chunk[-1]["role"] == "user":
                    print("  - Dropping trailing User message.")
                    current_chunk.pop()

                if len(current_chunk) > 1:
                    fixed_data.append({"messages": current_chunk})
                else:
                    print("  - Skipping too short conversation.")

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {i+1}")
                continue

        # Write back to file
        with open(input_file, "w", encoding="utf-8") as f:
            for entry in fixed_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(
            f"  -> Done. Orig: {original_count} | New: {len(fixed_data)} | Splits: {split_count}"
        )

    print("\nAll files processed.")


if __name__ == "__main__":
    split_long_conversations()

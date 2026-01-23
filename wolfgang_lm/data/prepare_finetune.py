import os
import glob
import json
import random
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

# Configuration
DATA_DIR = "data_clean"
TOKENIZER_FILE = os.path.join(DATA_DIR, "tokenizer.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "finetune_dataset.pt")

# Special Tokens (Must match train_tokenizer.py)
TOK_SYSTEM = "<|system|>"
TOK_USER = "<|user|>"
TOK_ASSISTANT = "<|assistant|>"
TOK_END = "<|endoftext|>"


def main():
    # 1. Load Tokenizer
    print(f"Loading tokenizer from {TOKENIZER_FILE}...")
    if not os.path.exists(TOKENIZER_FILE):
        print("Error: Tokenizer not found.")
        return
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

    # 2. Load Data
    jsonl_files = [
        os.path.join(DATA_DIR, "dataset_synthetic_conversational.jsonl"),
    ]
    print(
        f"Found {len(jsonl_files)} JSONL files: {[os.path.basename(f) for f in jsonl_files]}"
    )

    all_conversations = []
    for filepath in jsonl_files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "messages" in data:
                            all_conversations.append(data["messages"])
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line in {filepath}")

    print(f"Total conversations loaded: {len(all_conversations)}")

    if len(all_conversations) == 0:
        print("No conversations found. Exiting.")
        return

    # 3. Process and Tokenize each sample individually
    processed_samples = []
    skipped_long = 0
    MAX_SEQ_LEN = 512  # Should match model block_size

    print("Tokenizing and formatting samples...")
    for conv in tqdm(all_conversations):
        ids_list = []
        labels_list = []

        for msg in conv:
            role = msg.get("role")
            content = msg.get("content", "").strip()

            if role == "system":
                text = f"{TOK_SYSTEM}\n{content}\n"
                is_assistant = False
            elif role == "user":
                text = f"{TOK_USER}\n{content}\n"
                is_assistant = False
            elif role in ["assistant", "goethe"]:
                text = f"{TOK_ASSISTANT}\n{content}\n"
                is_assistant = True
            else:
                raise ValueError(f"Unknown role: {role}")
            chunk_ids = tokenizer.encode(text).ids

            # Append EOS token to every message
            eos_ids = tokenizer.encode(TOK_END).ids
            chunk_ids.extend(eos_ids)

            ids_list.extend(chunk_ids)

            # Create labels: -1 for non-assistant, copy ids for assistant
            if is_assistant:
                # Assistant: Train on content AND the EOS token
                labels_list.extend(chunk_ids)
            else:
                # User/System: Ignore everything (content + EOS)
                labels_list.extend([-1] * len(chunk_ids))

        # Check length
        if len(ids_list) > MAX_SEQ_LEN:
            skipped_long += 1
            # Optional: We could truncate here if we wanted, but let's just skip for high quality
            continue

        # Convert to tensors
        # We store them as list of tensors (or just plain lists reduced to tensors later)
        # Storing as plain lists in the dict is slightly more efficient for disk space if variable len
        processed_samples.append({"input_ids": ids_list, "labels": labels_list})

    print(f"Processed {len(processed_samples)} samples.")
    print(f"Skipped {skipped_long} samples > {MAX_SEQ_LEN} tokens.")

    # 4. Shuffle and Split
    random.seed(42)
    random.shuffle(processed_samples)

    split_idx = int(len(processed_samples) * 0.95)
    train_samples = processed_samples[:split_idx]
    val_samples = processed_samples[split_idx:]

    print(f"Split: {len(train_samples)} Train, {len(val_samples)} Validation")

    total_train_tokens = sum(len(s["input_ids"]) for s in train_samples)
    print(f"Total tokens in train set: {total_train_tokens}")

    # Lookup Padding Token
    pad_id = tokenizer.token_to_id("<|padding|>")
    if pad_id is None:
        print("Warning: <|padding|> token not found in tokenizer. Using 0.")
        pad_id = 0
    else:
        print(f"Found padding token ID: {pad_id}")

    # 5. Save
    print(f"Saving to {OUTPUT_FILE}...")
    torch.save(
        {
            "train": train_samples,
            "val": val_samples,
            "config": {"max_seq_len": MAX_SEQ_LEN, "pad_id": pad_id},
        },
        OUTPUT_FILE,
    )

    print("Data preparation complete.")


if __name__ == "__main__":
    main()

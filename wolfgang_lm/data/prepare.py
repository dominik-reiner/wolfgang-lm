import os
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

# Paths relative to project root
DATA_DIR = "data_clean"
INPUT_FILE = os.path.join(DATA_DIR, "corpus_pretrain.txt")
OUTPUT_FILE = os.path.join(DATA_DIR, "train.bin")
TOKENIZER_FILE = os.path.join(DATA_DIR, "tokenizer.json")


def prepare_data():
    if not os.path.exists(TOKENIZER_FILE):
        print("Error: Tokenizer not found.")
        return

    print(f"Loading tokenizer from {TOKENIZER_FILE}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

    print(f"Reading corpus {INPUT_FILE}...")

    # Check total size for progress bar estimate (optional but nice)
    total_size = os.path.getsize(INPUT_FILE)

    token_count = 0
    buffer = []
    BUFFER_SIZE = 1_000_000  # Flush every 1M tokens (~2MB RAM)

    with open(OUTPUT_FILE, "wb") as f_out:
        with open(INPUT_FILE, "r", encoding="utf-8") as f_in:
            pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Tokenizing")

            for line in f_in:
                pbar.update(len(line.encode("utf-8")))

                # We keep the newline structure implicit in the line read
                # Tokenizer handles it (usually map \n to space or specific token depending on training)
                # For this BPE, it processes the raw string 'line'
                encoded = tokenizer.encode(line)
                buffer.extend(encoded.ids)

                if len(buffer) >= BUFFER_SIZE:
                    arr = np.array(buffer, dtype=np.uint16)
                    f_out.write(arr.tobytes())
                    token_count += len(buffer)
                    buffer = []

            pbar.close()

            # Final flush
            if buffer:
                arr = np.array(buffer, dtype=np.uint16)
                f_out.write(arr.tobytes())
                token_count += len(buffer)

    print(f"\nData preparation complete.")
    print(f"Total Tokens: {token_count}")
    print(f"Binary file size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    prepare_data()

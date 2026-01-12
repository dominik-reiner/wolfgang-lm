import os
import glob

# Configuration

SOURCE_DIRS = [
    "data_clean/Belletristik_Core",
    "data_clean/Belletristik_Ext",
    "data_clean/Wissenschaft",
    "data_clean/Gebrauchsliteratur",
    "data_clean/Zeitung",
]
OUTPUT_FILE = "data_clean/corpus_pretrain.txt"
SEPARATOR = "\n<|endoftext|>\n"


def main():
    print("Concatenating files for pre-training corpus...")

    all_files = []
    for source_dir in SOURCE_DIRS:
        if os.path.exists(source_dir):
            files = sorted(glob.glob(os.path.join(source_dir, "*.txt")))
            print(f"Found {len(files)} files in {source_dir}")
            all_files.extend(files)
        else:
            print(f"Warning: Directory {source_dir} not found.")

    print(f"Total files: {len(all_files)}")

    all_content = []
    total_tokens_est = 0

    for filepath in all_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                all_content.append(content)
                # Rough estimate: 1 word ~= 1.3 tokens or just count chars/4
                total_tokens_est += len(content) / 4

    print(f"Joining {len(all_content)} documents...")

    # Join with the special EOS token
    full_corpus = SEPARATOR.join(all_content)

    # Add one at the very end too
    full_corpus += SEPARATOR

    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_corpus)

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"Success! Corpus size: {size_mb:.2f} MB")
    print(f"Estimated Tokens: {int(total_tokens_est):,}")


if __name__ == "__main__":
    main()

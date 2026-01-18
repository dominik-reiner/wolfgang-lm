import os
import json
import numpy as np
from tokenizers import Tokenizer


def analyze_dataset():
    tokenizer_path = "data_clean/tokenizer.json"
    files = [
        "data_clean/dataset_synthetic_conversational.jsonl",
    ]

    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        return

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # 512 context window
    BLOCK_SIZE = 512

    all_lengths = []
    outliers = []
    file_stats = {}

    for filepath in files:
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found.")
            continue

        print(f"Processing {filepath}...")
        lengths = []
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    messages = data.get("messages", [])

                    # Reconstruct the prompt exactly as in training/server
                    prompt = ""
                    for msg in messages:
                        role = msg.get("role")
                        content = msg.get("content")
                        if role == "system":
                            prompt += f"<|system|>\n{content}\n"
                        elif role == "user":
                            prompt += f"<|user|>\n{content}\n"
                        elif role == "assistant":
                            prompt += f"<|assistant|>\n{content}\n"

                    # Tokenize
                    encoded = tokenizer.encode(prompt)
                    length = len(encoded.ids) + 1  # +1 for EOS token <|endoftext|>
                    lengths.append(length)
                    all_lengths.append(length)

                    if length > BLOCK_SIZE:
                        outliers.append(
                            {
                                "file": os.path.basename(filepath),
                                "line": i + 1,
                                "length": length,
                                "excerpt": prompt[:100].replace("\n", " ") + "...",
                            }
                        )
                except json.JSONDecodeError:
                    print(f"Error decoding line {i+1} in {filepath}")

        file_stats[os.path.basename(filepath)] = lengths

    # Global Stats
    total_samples = len(all_lengths)
    if total_samples == 0:
        print("No data found.")
        return

    avg_len = np.mean(all_lengths)
    max_len = np.max(all_lengths)
    p95 = np.percentile(all_lengths, 95)
    p99 = np.percentile(all_lengths, 99)
    num_outliers = len(outliers)
    percent_outliers = (num_outliers / total_samples) * 100

    print("\n" + "=" * 50)
    print(f"FINAL ANALYSIS REPORT")
    print("=" * 50)
    print(f"Total Samples Analyzed: {total_samples}")
    print(f"Context Window: {BLOCK_SIZE} tokens")
    print(f"Average Length: {avg_len:.2f} tokens")
    print(f"Max Length: {max_len} tokens")
    print(f"95th Percentile: {p95:.2f} tokens")
    print(f"99th Percentile: {p99:.2f} tokens")
    print("-" * 30)
    print(f"OUTLIERS (> {BLOCK_SIZE} tokens):")
    print(f"Count: {num_outliers}")
    print(f"Percentage: {percent_outliers:.2f}%")
    print("-" * 30)

    print("\nPer-File Breakdown:")
    for filename, lens in file_stats.items():
        if not lens:
            continue
        avg = np.mean(lens)
        local_outliers = sum(1 for l in lens if l > BLOCK_SIZE)
        print(
            f"{filename: <30} | Avg: {avg:.1f} | Outliers: {local_outliers}/{len(lens)} ({local_outliers/len(lens)*100:.1f}%)"
        )

    print("\nTop 5 Longest Outliers:")
    outliers.sort(key=lambda x: x["length"], reverse=True)
    for o in outliers[:5]:
        print(
            f"- [{o['length']} tokens] {o['file']} (Line {o['line']}): {o['excerpt']}"
        )


if __name__ == "__main__":
    analyze_dataset()

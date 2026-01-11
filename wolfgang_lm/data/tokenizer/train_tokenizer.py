import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# Configuration
CORPUS_FILE = "data_clean/corpus_pretrain.txt"
OUTPUT_FILE = "data_clean/tokenizer.json"
VOCAB_SIZE = 32768  # Standard small-mid model size
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|padding|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


def train_tokenizer():
    print(f"Initializing BPE Tokenizer training on {CORPUS_FILE}...")

    # 1. Initialize a Byte-Level BPE Tokenizer
    # ByteLevel is crucial for GPT-2/GPT-3 style tokenization (handles all unicode bytes)
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 2. Configure Trainer
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,  # Eliminate single-use typos
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
    )

    # 3. Train
    if not os.path.exists(CORPUS_FILE):
        print(f"Error: Corpus file {CORPUS_FILE} not found.")
        return

    print("Training started... (this may take a minute)")
    tokenizer.train(files=[CORPUS_FILE], trainer=trainer)

    # 4. Save
    print(f"Saving tokenizer to {OUTPUT_FILE}...")
    tokenizer.save(OUTPUT_FILE)
    print("Tokenizer training complete.")

    # 5. Test
    sample_text = "Ihr naht euch wieder, schwankende Gestalten!"
    encoded = tokenizer.encode(sample_text)
    print(
        f"\nTest Encoding:\nInput: '{sample_text}'\nTokens: {encoded.tokens}\nIDs: {encoded.ids}"
    )


if __name__ == "__main__":
    train_tokenizer()

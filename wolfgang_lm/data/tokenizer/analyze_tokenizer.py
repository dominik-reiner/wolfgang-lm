from tokenizers import Tokenizer

TOKENIZER_FILE = "data_clean/tokenizer.json"


def analyze_tokenizer():
    print(f"Loading tokenizer from {TOKENIZER_FILE}...")
    try:
        tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    vocab = tokenizer.get_vocab()
    print(f"Vocabulary Size: {len(vocab)}")

    # 1. Check for Umlaute handling
    # Ideally, common umlaut words should be single tokens
    test_words = ["schön", "Glück", "Mädchen", "groß", "dass", "daß"]
    print("\n--- Umlaut & Special Char Check ---")
    for word in test_words:
        encoded = tokenizer.encode(word)
        print(f"'{word}' -> {encoded.tokens}")

    # 2. Check for Archaic vs Modern Spelling consistency
    # (Goethe uses "thun", "Theil", "Thüre")
    pairs = [("tun", "thun"), ("Teil", "Theil"), ("Tor", "Thor")]
    print("\n--- Archaic Spelling Check ---")
    for mod, arch in pairs:
        enc_mod = tokenizer.encode(mod).tokens
        enc_arch = tokenizer.encode(arch).tokens
        print(f"Modern: '{mod}' -> {enc_mod} | Archaic: '{arch}' -> {enc_arch}")

    # 3. Longest Tokens (often shows compound words or common phrases)
    sorted_vocab = sorted(vocab.items(), key=lambda x: len(x[0]), reverse=True)
    print("\n--- Top 20 Longest Tokens ---")
    # Filter out tokens that start with extremely weird chars if any, but usually BPE is fine
    # Note: 'Ġ' represents the proceeding space in RoBERTa/GPT-2 BPE
    for token, id in sorted_vocab[:20]:
        print(f"{token} (ID: {id})")

    # 4. Sample Sentence Encoding
    sample = "Edel sei der Mensch, hilfreich und gut! Denn das allein unterscheidet ihn von allen Wesen, die wir kennen."
    print(f"\n--- Sample Sentence ---\nInput: {sample}")
    encoded = tokenizer.encode(sample)
    print(f"Tokens: {encoded.tokens}")
    print(
        f"Token Count: {len(encoded.tokens)} (Chars: {len(sample)}) -> Compression: {len(sample)/len(encoded.tokens):.2f} chars/token"
    )


if __name__ == "__main__":
    analyze_tokenizer()

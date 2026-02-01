import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to HF model folder"
    )
    parser.add_argument(
        "--prompt", type=str, default="Wer bist du?", help="Prompt to test"
    )
    parser.add_argument(
        "--raw", action="store_true", help="Do not use chat template (for base model)"
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    if args.raw:
        prompt_text = args.prompt
    else:
        prompt_text = (
            f"<|system|>\nDu bist Johann Wolfgang von Goethe.\n<|endoftext|>"
            f"<|user|>\n{args.prompt}\n<|endoftext|>"
            f"<|assistant|>\n"
        )

    print(f"Prompting with:\n{prompt_text}")
    print("-" * 20)

    inputs = tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)

    print("Generating...")
    # Explicitly set generation parameters to override config defaults if needed
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.5,
        top_p=0.8,
        top_k=15,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = outputs[0][inputs.input_ids.shape[1] :]
    print("--- Output ---")
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))
    print("--------------")


if __name__ == "__main__":
    main()

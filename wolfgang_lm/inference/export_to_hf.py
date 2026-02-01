import argparse
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from wolfgang_lm.modeling.config import ModelConfig


def permute_rope_weights(w, n_head, head_dim):
    # w shape: (n_head * head_dim, input_dim)
    # Wolfgang (Complex/Interleaved): [a0, b0, a1, b1, ...] where (ai, bi) are pairs
    # Llama (Split/Rotated): [a0, a1, ..., b0, b1, ...] (first half real, second half imag components)

    # We permute such that:
    # Even indices (0, 2, 4...) move to First Half
    # Odd indices (1, 3, 5...) move to Second Half

    # Reshape to (n_heads, head_dim, input_dim)
    w = w.view(n_head, head_dim, -1)

    # Split even and odd rows within head_dim
    w_even = w[:, 0::2, :]
    w_odd = w[:, 1::2, :]

    # Concatenate [Even, Odd]
    w_new = torch.cat([w_even, w_odd], dim=1)

    # Flatten back
    return w_new.reshape(-1, w_new.shape[-1])


def convert_state_dict(wf_state_dict, config):
    new_sd = {}
    head_dim = config.n_embd // config.n_head

    # Remove _orig_mod. prefix if present (from torch.compile)
    wf_state_dict = {k.replace("_orig_mod.", ""): v for k, v in wf_state_dict.items()}

    # Mappings
    # Embeddings
    new_sd["model.embed_tokens.weight"] = wf_state_dict["embeddings.weight"]

    # Layers
    for i in range(config.n_layer):
        # Attention Norm - RMSNorm
        new_sd[f"model.layers.{i}.input_layernorm.weight"] = wf_state_dict[
            f"layers.{i}.attention_norm.weight"
        ]

        # Attention - Permute Q and K for RoPE compatibility
        # Wolfgang uses interleaved RoPE, Llama uses split RoPE

        q_weight = wf_state_dict[f"layers.{i}.attention.wq.weight"]
        k_weight = wf_state_dict[f"layers.{i}.attention.wk.weight"]

        new_sd[f"model.layers.{i}.self_attn.q_proj.weight"] = permute_rope_weights(
            q_weight, config.n_head, head_dim
        )
        new_sd[f"model.layers.{i}.self_attn.k_proj.weight"] = permute_rope_weights(
            k_weight, config.n_kv_head, head_dim
        )

        new_sd[f"model.layers.{i}.self_attn.v_proj.weight"] = wf_state_dict[
            f"layers.{i}.attention.wv.weight"
        ]
        new_sd[f"model.layers.{i}.self_attn.o_proj.weight"] = wf_state_dict[
            f"layers.{i}.attention.wo.weight"
        ]

        # Post Attention Norm - RMSNorm
        new_sd[f"model.layers.{i}.post_attention_layernorm.weight"] = wf_state_dict[
            f"layers.{i}.ffn_norm.weight"
        ]

        # MLP - SwiGLU
        # wolfgang: w1(gate), w2(down), w3(up/value)
        # llama: gate_proj(w1), down_proj(w2), up_proj(w3)
        new_sd[f"model.layers.{i}.mlp.gate_proj.weight"] = wf_state_dict[
            f"layers.{i}.feed_forward.w1.weight"
        ]
        new_sd[f"model.layers.{i}.mlp.up_proj.weight"] = wf_state_dict[
            f"layers.{i}.feed_forward.w3.weight"
        ]
        new_sd[f"model.layers.{i}.mlp.down_proj.weight"] = wf_state_dict[
            f"layers.{i}.feed_forward.w2.weight"
        ]

    # Final Norm
    new_sd["model.norm.weight"] = wf_state_dict["norm.weight"]

    # Output Head
    if "output.weight" in wf_state_dict:
        new_sd["lm_head.weight"] = wf_state_dict["output.weight"]
    else:
        # Weight tying: if output.weight is missing, reuse embeddings.weight
        print("Note: 'output.weight' not found in state_dict, sharing with embeddings.")
        new_sd["lm_head.weight"] = wf_state_dict["embeddings.weight"]

    return new_sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to wolfgang checkpoint (ckpt.pt)"
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Path to tokenizer.json"
    )
    parser.add_argument("--out", type=str, default="hf_export", help="Output directory")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location="cpu")

    # Load config
    mc_dict = checkpoint.get("model_config", {})
    default_mc = ModelConfig()
    for k, v in mc_dict.items():
        if hasattr(default_mc, k):
            setattr(default_mc, k, v)

    print(f"Model Config: {default_mc}")

    # Calculate intermediate size for SwiGLU
    # from MLP class: hidden_dim = 4 * n_embd; hidden_dim = int(2 * hidden_dim / 3)
    # hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    hidden_dim = 4 * default_mc.n_embd
    hidden_dim = int(2 * hidden_dim / 3)
    intermediate_size = default_mc.multiple_of * (
        (hidden_dim + default_mc.multiple_of - 1) // default_mc.multiple_of
    )

    # Create LlamaConfig
    hf_config = LlamaConfig(
        vocab_size=default_mc.vocab_size,
        hidden_size=default_mc.n_embd,
        intermediate_size=intermediate_size,
        num_hidden_layers=default_mc.n_layer,
        num_attention_heads=default_mc.n_head,
        num_key_value_heads=default_mc.n_kv_head,
        max_position_embeddings=default_mc.block_size,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=1,  # <|padding|>
        bos_token_id=None,
        eos_token_id=0,  # <|endoftext|>
        tie_word_embeddings=True,
        rope_theta=10000.0,
        # Generation Params Defaults
        temperature=0.05,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.15,
        do_sample=True,
    )

    print(f"HF Config: {hf_config}")

    print("Converting state dict...")
    model_sd = checkpoint["model"]
    hf_sd = convert_state_dict(model_sd, default_mc)

    print("Creating LlamaForCausalLM...")
    model = LlamaForCausalLM(hf_config)
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    print(f"Saving to {args.out}...")
    model.save_pretrained(args.out)

    # Tokenizer
    print("Saving tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    tokenizer.pad_token = "<|padding|>"
    tokenizer.eos_token = "<|endoftext|>"
    # Register other special tokens so they are preserved
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"]}
    )

    tokenizer.save_pretrained(args.out)

    print("Done. You can now use this folder with Hugging Face or convert to GGUF.")


if __name__ == "__main__":
    main()

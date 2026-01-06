from dataclasses import dataclass


@dataclass
class ModelConfig:
    block_size: int = 512  # Context window duration
    vocab_size: int = 32768  # Matches our BPE tokenizer
    n_layer: int = 12  # Transformer blocks
    n_head: int = 10  # Query heads
    n_kv_head: int = 5  # Key/Value heads (GQA with Group Size = 2)
    n_embd: int = 640  # Embedding dimension
    dropout: float = 0.0  # Modern LLMs often skip dropout for pretraining
    bias: bool = (
        False  # True: bias in Linears. False: a bit better/faster (Llama style)
    )
    multiple_of: int = 32  # Enforce SwiGLU hidden dim is multiple of this

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # -------------------------------------------------------------------------
    # System & Checkpoints
    # -------------------------------------------------------------------------
    out_dir: str = "out-pretrain"
    device: str = "cuda"  # 'cuda', 'cpu', 'mps'
    dtype: str = "bfloat16"  # 'float32', 'bfloat16', 'float16'
    compile: bool = True  # use PyTorch 2.0 torch.compile()

    # -------------------------------------------------------------------------
    # Data & Evaluation
    # -------------------------------------------------------------------------
    dataset: str = "data_clean"  # dataset directory
    eval_interval: int = 100  # validate every N steps
    log_interval: int = 10  # print log every N steps
    eval_iters: int = 200  # average N batches for validation
    always_save_checkpoint: bool = False

    # -------------------------------------------------------------------------
    # Optimization (Hyperparameters)
    # -------------------------------------------------------------------------
    # Batch Size: 32 sequences * 512 tokens = 16,384 tokens per micro-step
    batch_size: int = 32
    # Effective Batch Size = 32 * 8 = 256 sequences
    gradient_accumulation_steps: int = 8

    # Learning Rate Schedule (Cosine Decay)
    learning_rate: float = 6e-4  # max learning rate
    min_lr: float = 6e-5  # min learning rate (usually 10% of max)
    warmup_iters: int = 200  # linear warmup duration

    # Calculation:
    # Batch = 64 * 4 * 512 = 131,072 tokens/step
    # Dataset = ~254,000,000 tokens
    # Target = 4 Epochs (~1 Billion tokens)
    # Steps = (254M * 4) / 131k ≈ 7751 steps -> Round up to 8000
    max_iters: int = 8000
    lr_decay_iters: int = 8000  # match max_iters

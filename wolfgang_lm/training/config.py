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
    eval_interval: int = 250  # validate every N steps
    log_interval: int = 10  # print log every N steps
    eval_iters: int = 200  # average N batches for validation
    always_save_checkpoint: bool = True

    # -------------------------------------------------------------------------
    # Optimization (Hyperparameters)
    # -------------------------------------------------------------------------
    # Batch Size: 64 sequences * 512 tokens = 32,768 tokens per micro-step
    batch_size: int = 64
    # Effective Batch Size = 64 * 4 = 256 sequences
    gradient_accumulation_steps: int = 4

    # Learning Rate Schedule (Cosine Decay)
    learning_rate: float = 6e-4  # max learning rate
    min_lr: float = 6e-5  # min learning rate (usually 10% of max)
    warmup_iters: int = 500  # linear warmup duration
    max_iters: int = 10000  # total number of training steps
    lr_decay_iters: int = 10000  # should be ~= max_iters

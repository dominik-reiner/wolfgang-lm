import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter (Gamma). Allows the model to emphasize specific features.
        # This is a single vector of shape (dim,) that applies to all tokens via broadcasting.
        # Initialized to 1.
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Calculate RMS (Root Mean Square) over the last dimension (features).
        # We use .rsqrt() (1/sqrt) as it is computationally faster than sqrt() + div.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Mixed Precision Safety: Upcast to fp32 for precise norm calculation (avoid overflow/underflow),
        # then cast back to original type (fp16/bf16) for the rest of the network.
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # Calculate the base frequencies for each dimension pair.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    # Generate the angle for each position (t) and frequency: angle = t * freq
    freqs = torch.outer(t, freqs).float()
    # Create complex numbers: cos(angle) + i*sin(angle).
    # These act as the rotation factors when multiplied with the data.
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshapes the precomputed freq tensor to be broadcastable with the input x.
    freqs_cis: (Seq_Len, Head_Dim//2) -> 2D
    x: (Batch, Seq_Len, Heads, Head_Dim//2) -> 4D (Complex)
    Input x has dimensions [B, S, H, D]. We want to match dimension S and D.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # Create broadcast shape: [1, Seq_Len, 1, Head_Dim//2]
    # This aligns the sequence dimension and the feature dimension, while
    # letting Batch and Head dimensions broadcast (copy) the rotation.
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Applies Rotary Positional Embeddings (RoPE) by rotating the query and key vectors.
    """
    # 1. View as Complex:
    #    Reshape (B, S, H, D) -> (B, S, H, D/2, 2) -> (B, S, H, D/2) Complex.
    #    We pair adjacent floating point numbers to treat them as complex for
    #    rotation.
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 2. Reshape Frequencies: e.g. (S, 1, D/2)
    #    Ensure rotation matrix broadcasts over Batch and Heads.
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # 3. Apply Rotation: Multiply by complex frequency (equivalent to rotating the vector).
    # This uses the property that multiplying by e^(i*theta) rotates a complex number by angle theta.
    # (x + iy) * (cos(t) + i*sin(t)) results in the standard 2D rotation formula:
    # Real Part: x*cos(t) - y*sin(t)
    # Imag Part: x*sin(t) + y*cos(t)
    # The result is that every pair of numbers (x,y) in the head is rotated by the position-dependent angle.
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head

        # Project inputs to Query. We mix all input features to create the query vector for each head.
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        # Project K/V. If n_kv_head < n_head (GQA), these are smaller because multiple Q heads share one K/V head.
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        # Output Projection. Mixes the results from all attention heads back into the model's embedding space.
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

        # Store dropout probability. This is passed to the fused Flash Attention kernel,
        # which applies dropout efficiently on-the-fly during training without creating a mask tensor.
        self.dropout = config.dropout

    def forward(self, x, freqs_cis):
        # Batch, Sequence, Embedding
        B, S, E = x.shape
        # Project inputs to Query, Key, Value
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape for heads
        # 1. Unravel the embedding: (B, S, E) -> (B, S, n_head, head_dim)
        xq = xq.view(B, S, self.n_head, self.head_dim)
        xk = xk.view(B, S, self.n_kv_head, self.head_dim)
        xv = xv.view(B, S, self.n_kv_head, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # GQA: Repeat KV heads
        # Expect xk/xv shape: (B, S, n_kv_head, head_dim) -> (B, n_kv_head, S, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # If n_kv_head < n_head, we must repeat
        if self.n_rep > 1:
            xk = (
                xk[:, :, None, :, :]
                .expand(B, self.n_kv_head, self.n_rep, S, self.head_dim)
                .reshape(B, self.n_head, S, self.head_dim)
            )
            xv = (
                xv[:, :, None, :, :]
                .expand(B, self.n_kv_head, self.n_rep, S, self.head_dim)
                .reshape(B, self.n_head, S, self.head_dim)
            )

        # Flash Attention: computes Softmax(Q @ K.T) @ V efficiently.
        # y shape: (Batch, n_head, Sequence, head_dim)
        y = F.scaled_dot_product_attention(
            xq, xk, xv, dropout_p=self.dropout if self.training else 0, is_causal=True
        )

        # Reshape back to standard: (Batch, Sequence, n_head, head_dim) -> (Batch, Sequence, Embed_Dim)
        y = y.transpose(1, 2).contiguous().view(B, S, E)
        # Final projection: Mixes the results from all n_heads into a single embedding vector per token.
        return self.wo(y)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # SwiGLU hidden dimension calculation:
        # 1. Start with standard 4x expansion.
        hidden_dim = 4 * config.n_embd
        # 2. Scale by 2/3 because SwiGLU uses 3 projections instead of 2,
        #    so we reduce width to keep parameter count roughly constant
        #    (8/3 * d_model).
        hidden_dim = int(2 * hidden_dim / 3)
        # 3. Round up to the nearest multiple_of (e.g. 32) for hardware efficiency.
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # Gate
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # Value
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)  # Down

    def forward(self, x):
        # SwiGLU Forward Pass:
        # 1. Gate: w1(x) -> (B, S, Hidden)
        # 2. Value: w3(x) -> (B, S, Hidden)
        # 3. Activation: silu(Gate) * Value -> Element-wise multiplication
        # 4. Down: w2(...) -> (B, S, Embed)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.feed_forward = MLP(config)

    def forward(self, x, freqs_cis):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class WolfgangGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying
        self.output.weight = self.embeddings.weight

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size * 2,  # Reserve space for longer context inference later
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, S = idx.shape
        # 1. Token Embeddings: Convert indices to vectors (B, S, Embed_Dim)
        x = self.embeddings(idx)

        # 2. RoPE Prep: Select the rotation frequencies for this sequence length.
        #    We only need the first S rows from our precomputed table.
        freqs_cis = self.freqs_cis[:S]
        freqs_cis = freqs_cis.to(x.device)

        # 3. Transformer Block Stack:
        #    Each block applies Attention and MLP with residual connections.
        for layer in self.layers:
            x = layer(x, freqs_cis)

        # 4. Final Normalization: Apply RMSNorm before the output projection.
        x = self.norm(x)

        if targets is not None:
            # Training Mode: Calculate Loss
            # A. Compute logits for entire sequence (B, S, Vocab_Size)
            logits = self.output(x)
            # B. Flatten to (B*S, Vocab_Size) and compare with ground truth
            # Cross Entropy calculates -log(Probability_of_Correct_Token).
            # It severely penalizes the model if it assigns low probability to the actual next word.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Inference Mode: Generation
            # Optimization: Only compute logits for the very last token in the sequence.
            # We don't need predictions for past tokens during generation.
            # x[:, [-1], :] keeps dimension as (B, 1, Embed_Dim)
            logits = self.output(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # any weights that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # Weight Decay: A regularization technique that constantly shrinks parameter values towards 0.
        # This prevents overfitting by forcing the model to rely only on "strong" signals.
        # We skip it for 1D Parameters (biases/norms) because shrinking them could destroy model stability.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters"
        )

        # Use fused AdamW if available
        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )

        return optimizer

# Wolfgang-LM Architecture

Wolfgang-LM is a **75 Million Parameter** causal language model designed to emulate the style of Johann Wolfgang von Goethe.

Wolfgang-LM utilizes a **modern, Llama-style architecture** optimized for efficiency and training stability.

For details on the architecture, see [Architecture Deepdive](architecture_deep_dive.md).

## Technical Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Parameters** | **~75.5M** | Fits easily in consumer RAM/VRAM. |
| **Context Window** | **512** | Sufficient for paragraph-level generation and dialogue. |
| **Vocab Size** | **32,768** | Custom Byte-Pair Encoding (BPE). |
| **Layers** | **12** | Depth of the Transformer. |
| **Heads (Query)** | **10** | Number of attention heads. |
| **Heads (KV)** | **5** | **Grouped Query Attention (GQA)** with Factor 2. |
| **Embedding Dim** | **640** | Dimension of the latent space. |

## Modern Components

Wolfgang-LM is built from scratch in PyTorch 2.x and bypasses standard wrapper classes to implement state-of-the-art mechanisms:

### 1. Rotary Positional Embeddings (RoPE)
Instead of learning absolute position vectors, we apply relative positional information via rotation in the complex plane. This allows the model to generalize better to varying sequence lengths.

### 2. SwiGLU Activation
We replace the standard GELU activation with **Swish-Gated Linear Units**. This adds a learnable gating mechanism to the Feed-Forward Network, significantly improving perplexity for a given parameter budget.

### 3. RMSNorm (Root Mean Square Normalization)
We use RMSNorm instead of LayerNorm. It is computationally more efficient (no mean subtraction) and provides better training stability for deep networks.

### 4. Grouped Query Attention (GQA)
We use GQA (Group Size = 2). This reduces the size of the Key/Value cache during inference, making the model faster and less memory-intensive when generating text, without the quality degradation of Multi-Query Attention (MQA).

### 5. Flash Attention
The model is compatible with PyTorch's `F.scaled_dot_product_attention`, automatically utilizing CUDA-optimized Flash Attention kernels on supported hardware.

# Project Roadmap & TODOs

## Inference & Server
- [ ] **Implement KV Cache**:
  - Update `WolfgangGPT` to support caching of Key/Value states during generation to speed up inference (O(N) instead of O(N^2)).

# Future Roadmap: Wolfgang-LM V2 (Deep Reasoning Edition)
Moving to a **smaller vocabulary** and **more layers** to align with state-of-the-art research for small models (like Meta's *MobileLLM*). "Deep and Narrow" creates significantly smarter reasoning engines for the same parameter cost.

### 1. Strategy: Reallocate "Vocabulary Tax"
- **Current V1:** 32k Vocab, 12 Layers (Wide & Shallow).
- **Proposed V2:** 8k Vocab, 24 Layers (Deep & Narrow).
- **Math:** Reducing vocab from 32k to 8k saves ~18M parameters. Reducing dim from 640 to 384 saves more. Reinvest these into doubling the depth (12 -> 24 layers).

### 2. Architecture Spec
| Feature | V1 (Current) | **V2 (Deep & Narrow)** | Why V2? |
| --- | --- | --- | --- |
| **Layers** | 12 | **24** | **Reasoning Depth:** Logic requires serial steps. |
| **Embedding Dim** | 640 | **384** | **Efficiency:** divisible by 64/128. |
| **Heads (Q/KV)** | 10/5 | **6 / 3** | GQA Factor 2. |
| **Vocab Size** | 32,768 | **8,192** | **Focus:** Forces model to learn *concepts*. |
| **Context** | 512 | **1024** | Double context for "free" (memory-wise). |

### 3. Implementation Plan
- **New Tokenizer**: DO NOT just trim the old one. Train a **new BPE tokenizer** specifically on the 265M token dataset to find the most efficient 8,000 tokens for *this* specific text.
- **Config Changes**:
  ```python
  config = Config(
      vocab_size=8192,
      hidden_size=384,      # Narrow
      intermediate_size=1024, # SwiGLU
      num_hidden_layers=24, # DEEP!
      num_attention_heads=6,
      num_key_value_heads=3, # GQA
      max_position_embeddings=1024
  )
  ```

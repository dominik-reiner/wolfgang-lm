# Inference Generation Documentation

## Overview
This document explains the inference pipeline implemented in `wolfgang_lm.inference.generation.py`. The `WolfgangGenerator` class handles the entire generation process, from tokenizing the prompt to sampling logits and decoding the output stream.

## WolfgangGenerator Class

### Initialization
```python
generator = WolfgangGenerator(
    checkpoint_path="out-pretrain/ckpt_final.pt",
    tokenizer_path="data_clean/tokenizer.json",
    device="cuda" # or "mps", "cpu"
)
```
*   **Model Loading**: Loads the model weights and configuration from a saved checkpoint.
*   **State Dict Handling**: Automatically fixes key prefixes (e.g., removing `_orig_mod.`) if the model was compiled during training.
*   **Tokenizer**: Loads the BPE tokenizer for encoding/decoding.

## Sampling Parameters

The `generate` method supports a variety of parameters to control the creativity and coherence of the output:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | `str` | Required | The input text to continue. |
| `max_new_tokens` | `int` | `100` | Maximum number of tokens to generate. |
| `temperature` | `float` | `0.6` | Controls randomness. High (1.0+) is chaotic, Low (<0.5) is deterministic. |
| `top_k` | `int` | `40` | Limits sampling to the top `k` most likely tokens. |
| `top_p` | `float` | `0.9` | Nucleus Sampling. Cumulative probability threshold. |
| `repetition_penalty` | `float` | `1.2` | Penalizes tokens that have already appeared in the recent context. |
| `stop_tokens` | `list[int]` | `None` | List of token IDs that, if generated, stop the generation immediately. |
| `seed` | `int` | `None` | Random seed for reproducibility. |
| `stream` | `bool` | `False` | If True, returns a Python generator yielding text chunks as they are created. |

## Internal Logic

### 1. Tokenization
The prompt is encoded into token IDs using the BPE tokenizer.

### 2. The Sampling Loop
For each step up to `max_new_tokens`:
1.  **Context Slicing**: The model only sees the last `block_size` tokens (e.g., 512) to fit within its context window.
2.  **Forward Pass**: The model predicts logits for the next token.
3.  **Temperature Scaling**: Logits are divided by `temperature`.
4.  **Repetition Penalty**:
    - Implements the [CTRL](https://arxiv.org/abs/1909.05858) penalty.
    - If a token is in the recent context window (last 24 tokens), its logit is penalized (pushed towards negative infinity) to reduce likelihood.
5.  **Filtering**:
    - **Top-K**: Keeps only the K highest probability tokens.
    - **Top-P (Nucleus)**: Keeps the smallest set of tokens whose cumulative probability exceeds P.
6.  **Sampling**: A token is selected using `torch.multinomial`.

### 3. Streaming Decoding
When `stream=True`, the generator yields chunks of text.
*   **Unicode Stability**: Decoding BPE tokens one-by-one can result in broken characters (e.g., if a UTF-8 character is split across two tokens).
*   **Solution**: The decoder decodes the entire sequence at each step. If the end of the string contains a "replacement character" (indicating a split byte), it holds back that text until the next token completes the valid character.

## Usage Example

```python
from wolfgang_lm.inference.generation import WolfgangGenerator

gen = WolfgangGenerator("out-pretrain/ckpt_final.pt", "data_clean/tokenizer.json")

# Standard Generation
output = gen.generate("Faust: Das also war des Pudels Kern!", temperature=0.7)
print(output)

# Streaming Generation
stream = gen.generate("Gretchen:", stream=True)
for chunk in stream:
    print(chunk, end="", flush=True)
```

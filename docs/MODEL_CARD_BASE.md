---
license: mit
language:
- de
tags:
- historical
- literature
- dta
- text-generation
- llama
- pytorch
library_name: transformers
metrics:
- perplexity
---

# Wolfgang-LM (75M) - Base Model

**Wolfgang-LM** is a historical language model trained to replicate the linguistic style of late 18th and early 19th-century German literature. This is the **base pretrained model**, designed for historical text generation and as a foundation for style-specific fine-tuning.

> **⚠️ Research Preview**: This model is a research artifact for exploring historical linguistics and style transfer. It should not be used as a source of factual information.

## Model Details

*   **Model Type**: Decoder-Only Transformer (Llama-style architecture)
*   **Parameters**: ~75M
*   **Language(s)**: German (Historical: 1600-1900)
*   **Context Window**: 512 Tokens
*   **License**: MIT License
*   **Finetuned from model**: None (Trained from scratch)

## Uses

### Intended Use
*   **Historical Text Generation**: Generating text in the style of 18th/19th-century German literature.
*   **Style Transfer Foundation**: Serving as a base model for fine-tuning specific historical personas (e.g., Goethe, Schiller).
*   **Linguistic Analysis**: Exploring grammatical structures and vocabulary of the period.

### Out-of-Scope Use
*   **Instruction Following**: This is a pure text completion model. It will not follow instructions, answer questions, or act as an assistant (use `dominik-reiner/wolfgang-lm-75M-instruct-v1` for this).
*   **Modern Context**: The model was trained exclusively on historical data. It does not understand modern concepts, technology, or current events.
*   **Factual Accuracy**: The model generates plausible-sounding historical text but should not be used as a fact-checking tool.

## Training Data

The model was trained on a strictly curated corpus to ensure historical authenticity, sourced from the **Deutsches Textarchiv (DTA)**.

1.  **Deutsches Textarchiv (DTA) Core Corpus**
    *   **Attribution**: Sourced from the [Deutsches Textarchiv](https://www.deutschestextarchiv.de/), Berlin-Brandenburg Academy of Sciences and Humanities.
    *   **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
    *   **Content**:
        *   *Belletristik* (Fiction - Core & Extended)
        *   *Wissenschaft* (Science)
        *   *Gebrauchsliteratur* (Utilitarian texts)
        *   *Zeitung* (Historical Newspapers)

## Evaluation

*   **Validation Loss (Cross Entropy)**: ~ 3.9
*   **Perplexity**: ~ 50

## Bias, Risks, and Limitations

*   **Historical Bias**: The model reflects the social, gender, and scientific biases prevalent in 18th/19th-century Germany.
*   **Hallucination**: As a small model (75M), it frequently hallucinates facts and generates plausible-sounding but nonsensical text.

## How to Get Started

This model can be used with the `transformers` library.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "dominik-reiner/wolfgang-lm-75M-base-v1"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Die Natur ist"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(
    **inputs, 
    max_new_tokens=100,
    temperature=0.5,
    top_p=0.8,
    top_k=15,
    repetition_penalty=1.1,
    do_sample=True
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Technical Specifications

*   **Tokenizer**: Byte-Pair Encoding (BPE), vocab size 32,768.
*   **Position Embeddings**: Rotary Positional Embeddings (RoPE).
*   **Attention**: Grouped Query Attention (GQA) with 10 heads (5 KV heads).
*   **Activation**: SwiGLU.
*   **Weight Tying**: Input embeddings and output head share weights.

## Resources

*   **GitHub Repository**: [dominik-reiner/wolfgang-lm](https://github.com/dominik-reiner/wolfgang-lm)
*   **Architecture**: [Model Architecture](https://github.com/dominik-reiner/wolfgang-lm/blob/main/docs/model_architecture.md)
*   **Data Sources**: [Data Documentation](https://github.com/dominik-reiner/wolfgang-lm/blob/main/docs/data_sources.md)
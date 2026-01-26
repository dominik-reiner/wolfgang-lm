---
license: mit
language:
- de
tags:
- historical
- literature
- goethe
- text-generation
base_model: null
datasets:
- deutschestextarchiv/dta
---

# Wolfgang-LM (75M) - Model Card

**Wolfgang-LM** is a historical language model trained to replicate the linguistic style of late 18th and early 19th-century German literature, specifically emulating the persona of **Johann Wolfgang von Goethe**.

> **⚠️ Research Preview**: This model is a research artifact for exploring historical linguistics and style transfer. It should not be used as a source of factual information.

## Model Details

*   **Developed by**: [Your Name/Handle]
*   **Model Type**: Decoder-Only Transformer (Llama-style architecture)
*   **Parameters**: ~75M
*   **Language(s)**: German (Historical: 1750-1832)
*   **License**: MIT License
*   **Finetuned from model**: None (Trained from scratch)

## Uses

### Intended Use
*   **Historical Simulation**: Simulating conversations with a Goethe-like persona.
*   **Style Transfer**: Rewriting modern German text into Weimar Classicism style.
*   **Education**: Exploring grammatical structures of the 19th century.

### Out-of-Scope Use
*   **Factual Queries**: The model has no knowledge of events after ~1832 and hallucinates historically.
*   **Modern Assistance**: Not suitable for programming, math, or current events.

## Training Data

The model was trained on a strictly curated corpus to ensure historical authenticity.

1.  **Deutsches Textarchiv (DTA) Core Corpus**
    *   **Attribution**: Sourced from the [Deutsches Textarchiv](https://www.deutschestextarchiv.de/), Berlin-Brandenburg Academy of Sciences and Humanities.
    *   **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
    *   **Content**: Belletristik, Wissenschaft, Gebrauchsliteratur, and Newspapers (17th-19th Century).
2.  **Conversational Data**
    *   *Conversations with Goethe* (J.P. Eckermann) - Public Domain.
    *   **Synthetic Data**: Generated using Google Gemini to facilitate style transfer for modern concepts.

## Bias, Risks, and Limitations

*   **Historical Bias**: The model reflects the social, gender, and scientific biases prevalent in 18th/19th-century Germany.
*   **Hallucination**: As a small model (75M), it frequently hallucinates facts.

## Compliance (EU AI Act)

*   **Transparency**: This is an Artificial Intelligence system.
*   **Risk Classification**: This model is a specific-purpose research artifact and does not classify as a "General Purpose AI Model with Systemic Risk" (>10^25 FLOPs).
*   **Generative AI Terms**: Training on synthetic data from Gemini follows the Google Generative AI Terms of Service (non-compete for general-purpose models). This model is a narrow, non-competing historical persona.

## How to Get Started

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/wolfgang-lm")
tokenizer = AutoTokenizer.from_pretrained("your-username/wolfgang-lm")

prompt = "Was denkst du über die Natur?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

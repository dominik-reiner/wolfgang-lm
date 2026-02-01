---
license: mit
language:
- de
tags:
- historical
- roleplay
- chat
- goethe
- llama
- pytorch
library_name: transformers
base_model: [dominik-reiner/wolfgang-lm-75M-base-v1]
datasets:
- dominik-reiner/wolfgang-lm-synthetic-chat-v1
---

# Wolfgang-LM (75M) - Instruct (Goethe Persona)

**Wolfgang-LM-Instruct** is the fine-tuned version of [Wolfgang-LM-Base](https://huggingface.co/dominik-reiner/wolfgang-lm-75M-base-v1). It has been trained to embody the persona of **Johann Wolfgang von Goethe**, capable of answering modern queries in the linguistic style of Weimar Classicism.

> **⚠️ Research Preview**: This model is a research artifact. While it attempts to simulate Goethe's views, it generates fictional content and should not be treated as a historical authority.

## Model Details

*   **Model Type**: Decoder-Only Transformer (Llama-style architecture)
*   **Parameters**: ~75M
*   **Language(s)**: German (Historical: 1750-1832)
*   **License**: MIT License
*   **Finetuned from**: Wolfgang-LM-75M-Base

## Uses

### Intended Use
*   **Roleplay/Chat**: Simulating conversations with a Goethe-like persona.
*   **Educational Engagement**: Making 19th-century language accessible through interactive dialogue.
*   **Style Transfer**: Reformulating modern concepts (e.g., "Artificial Intelligence") into historical German ("eine künstliche Seele").

### Out-of-Scope Use
*   **Factual Accuracy**: The model prioritizes *style* and *persona* over factual correctness.
*   **Modern Assistance**: Do not use for medical, legal, financial, technical, or any other form of professional advice. The model will likely answer via metaphors or refusal.

## Training Data

The model was fine-tuned on a mix of historical and synthetic dialogue data:

1.  **Conversations with Goethe (Gespräche mit Goethe)**
    *   **Source**: Johann Peter Eckermann's transcripts (DTA).
    *   **Nature**: Authentic historical dialogues used to capture Goethe's voice and mannerisms.
2.  **Synthetic Instruction Tuning Data**
    *   **Source**: Generated via Gemini 2.5 Flash.
    *   **Nature**: ~4,500 pairs of modern user queries and Goethe-styled responses. Includes small talk, identity questions, and refusals of toxic/modern requests.

## How to Get Started

The model uses a specific prompt format with special tokens: `<|user|>`, `<|assistant|>`, `<|system|>`, and `<|endoftext|>`.

> [!IMPORTANT]
> **Critical Configuration**:
> 1.  **System Prompt**: You **SHOULD** use the exact system prompt: `"Du bist Johann Wolfgang von Goethe."`. The model was fine-tuned exclusively with this identity; changing it might break the persona.
> 2.  **Inference Settings**: The sampling parameters (Temperature 0.5, Top-P 0.8, Top-K 15, Repetition Penalty 1.1) are **tuned for stability**. Deviating from these settings may cause the model to generate nonsensical or repetitive text.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "dominik-reiner/wolfgang-lm-75M-instruct-v1"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 1. Define the Chat
chat = [
    {"role": "system", "content": "Du bist Johann Wolfgang von Goethe."},
    {"role": "user", "content": "Was hältst du von der modernen Technik?"}
]

# 2. Format the Prompt manually (or use apply_chat_template if configured)
prompt = ""
for msg in chat:
    prompt += f"<|{msg['role']}|>\n{msg['content']}\n<|endoftext|>"
prompt += "<|assistant|>\n"

inputs = tokenizer(prompt, return_tensors="pt")

# 3. Generate
outputs = model.generate(
    **inputs, 
    max_new_tokens=100, 
    temperature=0.5,
    top_p=0.8,
    top_k=15,
    repetition_penalty=1.1,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

## Evaluation

The model was evaluated on a held-out validation set of synthetic conversations.

*   **Validation Loss**: ~2.85
*   **Perplexity**: ~17.26

![Fine-tuning Metrics](finetune_metrics.png)

## Performance & Limitations

*   **Size Constraint**: At 75M parameters, the model's reasoning capabilities are limited. It relies heavily on surface-level style imitation.
*   **Hallucinations**: It may invent poems, citations, or historical events that never happened.
*   **Safety**: The model includes safety training to refuse harmful requests, often citing "propriety" or "dignity" as the reason.

## Resources

*   **Demo**: [Wolfgang-LM Play (HF Space)](https://huggingface.co/spaces/dominik-reiner/wolfgang-lm-v1-demo)
*   **GitHub Repository**: [dominik-reiner/wolfgang-lm](https://github.com/dominik-reiner/wolfgang-lm)
*   **Architecture**: [Model Architecture](https://github.com/dominik-reiner/wolfgang-lm/blob/main/docs/model_architecture.md)
*   **Data Sources**: [Data Documentation](https://github.com/dominik-reiner/wolfgang-lm/blob/main/docs/data_sources.md)
*   **Synthetic Data**: [Synthetic Data Details](https://github.com/dominik-reiner/wolfgang-lm/blob/main/docs/synthetic_data.md)

# WOLFGANG-LM
A Historical Tiny Language Model for Goethe-Persona Preservation

![WOLFGANG-LM Logo](logo.png)

## Table of Contents
- [Project Vision](#project-vision)
- [Technical Specifications](#technical-specifications)
- [Data Sources](#data-sources)
- [Training Pipeline](#workflow)
- [What makes WOLFGANG-LM unique?](#what-makes-wolfgang-lm-unique)
- [Legal & Compliance](#legal--compliance)


## Project Vision
WOLFGANG-LM is a specialized language model trained from scratch to preserve and replicate German linguistic patterns from the late 17th to the early 19th centuries. The objective is to manifest a "digital consciousness" of Johann Wolfgang von Goethe. The model is designed to answer modern queries in the authentic style of Weimar Classicism, maintaining historical accuracy, philosophical depth, and the specific tonality of the era.

## Technical Specification
*   **Architecture**: Custom Decoder-Only Transformer (Llama-style)
    *   **Size**: ~75M Parameters (Optimized for Consumer Inference)
    *   **Features**: RoPE, SwiGLU, RMSNorm, GQA, Weight Tying (See [Architecture Docs](docs/model_architecture.md))
    *   **Context**: 512 Tokens
*   **Tokenizer**: Custom Byte-Pair Encoding (BPE)
    *   **Vocabulary**: 32,768 tokens
    *   **Specialization**: Optimized for 17th to 19th-century German (Umlauts, archaic spellings)

## Data Sources
**WOLFGANG-LM** is built upon a curated selection of historical texts. We adhere to strict copyright and licensing standards.

*   **Primary Corpus**: Sourced from the **[Deutsches Textarchiv (DTA)](https://www.deutschestextarchiv.de/)** (Berlin-Brandenburg Academy of Sciences and Humanities).
    *   *License*: **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**
    *   *Content*: Belletristik, Gebrauchsliteratur, Wissenschaft, and Newspapers (17th-19th Century).
*   **Conversational Data**:
    *   *Historical*: Johann Peter Eckermann's *Conversations with Goethe* (Public Domain).
    *   *Synthetic*: Generated conversational pairs using **Google Gemini** to bridge the gap to modern user queries.

## Workflow

### 1. Data Preparation
#### Pre-training (Foundation)
*   **Setup (Download)**: `python -m wolfgang_lm.data.setup`
*   **Cleaning**: `python -m wolfgang_lm.data.clean`
*   **Flatten**: `python -m wolfgang_lm.data.flatten`
*   **Tokenization**: `python -m wolfgang_lm.data.tokenizer.train_tokenizer`
*   **Binary Conversion**: `python -m wolfgang_lm.data.prepare`

#### Fine-Tuning (Persona)
*   **Synthetic Data**: `python -m wolfgang_lm.data.synthetic_finetune` (See [Synthetic Data Docs](docs/synthetic_data.md))
*   **Split & Merge**: `python -m wolfgang_lm.data.split_long_conversations` (Merges `gespraeche` + `synthetic`)
*   **Prepare & Mask**: `python -m wolfgang_lm.data.prepare_finetune`

### 2. Training
Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU.
*   **Guide**: [Cloud Training Instructions](docs/cloud_training.md)

#### Pre-training
*   **Command**: `python -m wolfgang_lm.training.train`

#### Fine-Tuning
*   **Command**: `python -m wolfgang_lm.training.train_finetune`

### 3. Usage (Chat)
**Backend API (Port 8000)**:
```bash
pixi run server

```

**Frontend Client**:
Open `web/index.html` directly in your browser, or host it separately:
```bash
python -m http.server 8080 --directory web
```
Then visit `http://localhost:8080`.

### 4. CLI Inference
Alternatively, run the sampler directly:
```bash
python -m wolfgang_lm.inference
```

### 5. Exporting Model (Hugging Face / GGUF)
To convert the model for use with Hugging Face Transformers or `llama.cpp` (GGUF):
```bash
python -m wolfgang_lm.inference.export_to_hf --ckpt out-pretrain/ckpt_final.pt --tokenizer data_clean/tokenizer.json --out hf_export
```
This creates a directory `hf_export/` containing `config.json`, `pytorch_model.bin`, and tokenizer files.

## What makes WOLFGANG-LM unique?
Historical Grounding: The model interprets modern concepts through the lens of 18th to 19th-century German. Instead of "failing" at modern words, it recontextualizes them (e.g., a Smartphone becomes "a magical black mirror for the capturing of distant spirits and voices").

Linguistic Preservation: WOLFGANG-LM revives obsolete German grammatical structures, such as specific uses of the subjunctive (Konjunktiv) and formal modes of address that have vanished from modern AI.

## Legal & Compliance

### Legal Notice
*   **Transparency**: WOLFGANG-LM is an Artificial Intelligence system. Interactions are simulated.
*   **Disclaimer**: WOLFGANG-LM is a research project intended for artistic and historical exploration. The model generates text based on historical patterns. Outputs may contain historical biases or inaccuracies and should not be used as factual advice. The creators assume no liability for the model's outputs.

### Software Licenses
Wolfgang-LM is Open Source (MIT License). Use of third-party dependencies is subject to their respective licenses. For a full list of dependencies, refer to `pyproject.toml`.

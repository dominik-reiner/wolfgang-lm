# Fine-Tuning Data Preparation

## Overview
This document details the specialized processing pipeline used to create the fine-tuning dataset for WOLFGANG-LM. Unlike the pre-training data which is ingested as raw text, the fine-tuning data requires conversion from a narrative format into a structured dialogue (chat) format.

## Goal
The source material, *Gespräche mit Goethe* by Johann Peter Eckermann, is a narrative work where Eckermann describes his conversations with Goethe.
*   **Original**: "I asked Goethe about his opinion on Schiller. He replied that..."
*   **Target**: 
    *   **User**: "What is your opinion on Schiller?"
    *   **Assistant**: [Goethe's actual reply]

## Implementation
The extraction is performed using the script `wolfgang_lm/data/extract_dialogue.py`.

### 1. Chunking Strategy
The raw text files are split into large chunks (default `50,000` characters) to fit within the context window of the extraction model. A smart overlapping strategy (default `1,000` characters) is used to ensure conversation threads that cross chunk boundaries are not lost or cut abruptly.

### 2. LLM-Based Extraction
We utilize **Google Gemini** (`gemini-2.5-flash` / `gemini-3-flash-preview`) to perform the semantic transformation. 

**System Prompt Strategy:**
The model is instructed to:
1.  **Identify Multi-Turn Dialogues**: Capture complete conversation flows (Question -> Answer -> Follow-up).
2.  **Narrative to Dialogue**: Rewrite Eckermann's indirect reporting ("I said") into direct speech for the "User" role.
3.  **Preserve Voice**: Maintain Goethe's original wording exactly as the "Assistant" response.
4.  **Format**: return structured JSON.

### 3. Post-Processing & Formatting
The extracted threads are saved in JSONL format suitable for fine-tuning.
*   **System Message**: Every conversation thread is prepended with a system prompt: *“Du bist Johann Wolfgang von Goethe. Antworte im Stil der Weimarer Klassik.”*
*   **Output File**: `data_clean/gespraeche.jsonl`

## Execution
To run the extraction pipeline:
```bash
# Requires GEMINI_API_KEY environment variable
python -m wolfgang_lm.data.extract_dialogue
```

## See Also
For generating synthetic training data (modern concepts, style bridging), see [Synthetic Data Generation](synthetic_data.md).

## 4. Dataset Composition (Automatic Aggregation)

The fine-tuning preparation script is designed to **automatically detect and aggregate all `.jsonl` files** found in the `data_clean/` directory.

Current Dataset Components:
1.  **`dataset_synthetic_conversational.jsonl`**: The unified synthetic dataset containing all modern, safety, and creative samples (~3500 samples).

> **Note**: The historical `gespraeche.jsonl` is currently generated for reference/style injection but not directly included in the training set in the current configuration.

## 5. Verification Report: Dataset Size & Quality

Based on the model architecture (**75M Parameters**) and current research, the dataset size of **~1,277 conversations** is mathematically ideal for this specific task.

### A. The "LIMA" Ratio (Quality > Quantity)
The 2023 [LIMA Paper](https://arxiv.org/abs/2305.11206) ("Less Is More for Alignment") demonstrated that for Supervised Fine-Tuning (SFT), the *quantity* of data is far less important than the *quality* and *diversity*.

*   **LIMA Standard:** 1,000 diverse, high-quality prompts were enough to align a 65B model.
*   **Our Dataset:** **~3,500 samples**.
*   **Conclusion:** We have hit the "LIMA benchmark". Adding significantly more synthetic/filler data would likely degrade performance by diluting the specific "Goethe" signal with generic AI patterns.

### B. Surface Alignment Mechanics
Fine-tuning does **not** teach the model new facts/logic (that happened in Pre-training). It teaches the model **how to interact** (formatting, tone, persona).

### C. Distribution Analysis

| Category | Count (Approx.) | Impact | Purpose |
| :--- | :--- | :--- | :--- |
| **Small Talk / Functional** | ~2800 | High (Generalization) | Covers modern concepts, everyday chat, and "style bridging" where Goethe encounters slang. |
| **Task Refusal** | ~350 | High (Character) | Teaches Goethe to refuse being a tool/assistant ("I am a poet, not a servant"). |
| **Safety** | ~350 | Critical (Constraints) | Handling toxicity or illegal requests with dignity. |

**Assessment:** The dataset is dominated by synthetic interactions (~3500) to enforce the persona in modern contexts (Small Talk, Refusal, Safety).

## 6. Prepare Fine-Tuning Data
Run the preparation script. It will scan `data_clean/` for the configured `.jsonl` files (currently only synthetic), tokenize them, and create a PyTorch dataset with proper masking (training only on Assistant responses).

```bash
pixi run python -m wolfgang_lm.data.prepare_finetune
```
*   **Output**: `data_clean/finetune_dataset.pt` (Contains `train` and `val` lists with input_ids and labels).

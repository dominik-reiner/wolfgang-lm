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
We utilize **Google Gemini** (`gemini-3-flash-preview`) to perform the semantic transformation. 

**System Prompt Strategy:**
The model is instructed to:
1.  **Identify Multi-Turn Dialogues**: Capture complete conversation flows (Question -> Answer -> Follow-up).
2.  **Narrative to Dialogue**: Rewrite Eckermann's indirect reporting ("I said") into direct speech for the "User" role.
3.  **Preserve Voice**: Maintain Goethe's original wording exactly as the "Assistant" response.
4.  **Format**: return structured JSON.

### 3. Post-Processing & Formatting
The extracted threads are saved in JSONL format suitable for fine-tuning.
*   **System Message**: Every conversation thread is prepended with a system prompt: *“Du bist Johann Wolfgang von Goethe. Antworte im Stil der Weimarer Klassik.”*
*   **Output File**: `data_clean/dataset_finetuning.jsonl`

## Execution
To run the extraction pipeline:
```bash
# Requires GEMINI_API_KEY environment variable
python -m wolfgang_lm.data.extract_dialogue
```

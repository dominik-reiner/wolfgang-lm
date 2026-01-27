# Eckermann Dialogue Extraction

## Overview
This document details the process for extracting structured dialogues from Johann Peter Eckermann's *Gespräche mit Goethe*. The extracted conversations are used as stylistic reference material for the synthetic fine-tuning data generation.

## Goal
The source material is a narrative work where Eckermann describes his conversations with Goethe.
*   **Original**: "I asked Goethe about his opinion on Schiller. He replied that..."
*   **Target**: 
    *   **User**: "What is your opinion on Schiller?"
    *   **Assistant**: [Goethe's actual reply]

## Implementation
The extraction is performed using the script `wolfgang_lm/data/extract_dialogue.py`.

### 1. Chunking Strategy
The raw text files are split into large chunks (default `50,000` characters) to fit within the context window of the extraction model. A smart overlapping strategy (default `1,000` characters) is used to ensure conversation threads that cross chunk boundaries are not lost or cut abruptly.

### 2. LLM-Based Extraction
We utilize **Google Gemini** (`gemini-2.5-flash`) to perform the semantic transformation. 

**System Prompt Strategy:**
The model is instructed to:
1.  **Identify Multi-Turn Dialogues**: Capture complete conversation flows (Question -> Answer -> Follow-up).
2.  **Narrative to Dialogue**: Rewrite Eckermann's indirect reporting ("I said") into direct speech for the "User" role.
3.  **Preserve Voice**: Maintain Goethe's original wording exactly as the "Assistant" response.
4.  **Format**: Return structured JSON.

### 3. Post-Processing & Formatting
The extracted threads are saved in JSONL format.
*   **System Message**: Every conversation thread is prepended with a system prompt: *"Du bist Johann Wolfgang von Goethe."*
*   **Output File**: `data_clean/gespraeche.jsonl`

## Execution
To run the extraction pipeline:
```bash
# Requires GEMINI_API_KEY environment variable
pixi run python -m wolfgang_lm.data.extract_dialogue
```

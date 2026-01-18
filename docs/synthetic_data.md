# Synthetic Fine-Tuning Data Generation

To prepare **Wolfgang-LM** for modern chatbot interactions, we use synthetic data generation to bridge the gap between 19th-century literature and 21st-century users.

## Overview

The script `wolfgang_lm/data/synthetic_finetune.py` uses a teacher LLM (**Gemini 3 Flash**) to generate a rich conversational dataset. All system prompts are distinctively **German** and use real Goethe dialogues (`gespraeche.jsonl`) as style references.

1.  **Modern Concepts**: Explaining ~50 modern topics (Bitcoin, Veganism) with Goethe's worldview.
2.  **Style Bridge**: Handling slang ("Digga", "Cringe") with dignity.
3.  **Identity & Safety**: Robustly refusing unsafe requests while staying in character.
4.  **Personal & Creative**: Answering bio questions details and writing poems/raps.
5.  **Philosophy & Sage**: Deep dives on timeless human issues.

## Features
- **German Prompts**: Native German instructions for better linguistic nuance.
- **Robustness**: Built-in exponential backoff for API rate limits.
- **Diversity**: High temperature (0.9) to ensure varied phrasing.
- **Comprehensive**: Expanded hardcoded input lists for broad coverage.

## Setup

Ensure you have a valid Google Gemini API Key.
Create a `.env` file in the project root if you haven't already:

```bash
GEMINI_API_KEY=your_api_key_here
```

## Usage

Run the generation script:

```bash
pixi run python -m wolfgang_lm.data.synthetic_finetune
```

The script will:
1.  Connect to the Gemini API.
2.  Generate responses for expanded categories (Casual, Modern, Sage, Philosophy, Safety, Personal, Creative).
3.  **Inject Real Style**: Samples real Goethe dialogues for authentic tonality.
4.  **Vary User Persona**: Simulates 5 different user styles (Slang, Academic, etc.).
5.  Save all results to `data_clean/dataset_synthetic_conversational.jsonl`.

## Integration

After generating this file, run the split script to merge and prepare it for training:

```bash
pixi run python -m wolfgang_lm.data.split_long_conversations
```

# Data Sources Documentation

## Overview
This document details the datasets acquired and used for the WOLFGANG-LM project. The data is divided into two primary categories: **Foundation Data** (General Language) and **Persona Data** (Fine-tuning).

## 1. Foundation Dataset
*   **Purpose**: Used for the Pre-training phase (Causal Language Modeling).
*   **Content**: A curated collection of historical German texts from the period 1600–1900.
*   **Source**: **DTA Normalized Corpus (2020-10-23)**. Downloaded directly via `python -m wolfgang_lm.data.setup`.
*   **Size (Cleaned)**: ~980 MB
*   **Tokens**: ~254 Million (Raw) -> ~1.01 Billion (Effective Training with 4 Epochs)
*   **File Count**: 3,566 Text Files
*   **Directories**:
    *   `data/Belletristik_Core` (Fiction - Core Selection)
    *   `data/Belletristik_Ext` (Fiction - Extended Selection)
    *   `data/Wissenschaft` (Science)
    *   `data/Gebrauchsliteratur` (Utilitarian)
    *   `data/Zeitung` (Newspapers)
*   **Characteristics**:
    *   High-fidelity literary and formal German.
    *   Includes archaic spellings and typography.
    *   Rich in philosophical, poetic, and narrative structures.

## 2. Persona Dataset (`data/gespraeche`)
*   **Purpose**: Used as a **Style Reference** for synthetic data generation and potential future fine-tuning.
*   **Content**: Eckermann's "Conversations with Goethe" (*Gespräche mit Goethe*).
*   **Source**: **DTA Normalized Corpus (2020-10-23)**. Downloaded directly via `python -m wolfgang_lm.data.setup`.
*   **Files**:
    *   `eckermann_goethe01_*.txt` (Part 1)
    *   `eckermann_goethe02_*.txt` (Part 2)
    *   `eckermann_goethe03_*.txt` (Part 3)
*   **Characteristics**:
    *   Dialogue-heavy format.
    *   Captures Goethe's conversational voice, opinions, and mannerisms.
    *   Contains distinct "Question/Answer" or "Stimulus/Response" structures suitable for training a chatbot persona.
    *   **Processing**: Narrative text was converted into dialogue format using `wolfgang_lm/data/extract_dialogue.py` (Gemini 2.5 Flash Preview). See [Fine-Tuning Preparation](fine_tuning_preparation.md) for more details.

## 3. Synthetic Data (The "Bridge")
*   **Purpose**: Used to teach the model how to discuss generic modern topics (Internet, AI) using its 19th-century persona.
*   **Source**: Generated via **Gemini 2.5 Flash** using the script `wolfgang_lm/data/synthetic_finetune.py`.
*   **Size**: ~4,500 High-Quality Samples.
*   **Content Distribution**:
    *   **Small Talk (60%)**: Casual conversations, modern concepts, style bridging (slang → dignified text).
    *   **Identity (20%)**: Biographical knowledge, questions about Goethe's life, works, and history.
    *   **Task Refusal (10%)**: Goethe refuses to act as a tool/assistant.
    *   **Safety (10%)**: In-character refusals of harmful/toxic requests.

See [Synthetic Data](synthetic_data.md) for more details.

## 4. Data Lineage & Changes
*   **Original Location**: Files were downloaded and split into `data/Belletristik_Core` and `data/Belletristik_Ext` (plus other categories).
*   **Restructuring**: Identified Eckermann files within `Belletristik` and moved them to a dedicated `data/gespraeche` directory to ensure clean separation between pre-training and fine-tuning data.

## 5. Licensing

*   **DTA Corpus (Foundation & Persona)**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
    *   **Attribution**: Berlin-Brandenburgische Akademie der Wissenschaften. Deutsches Textarchiv

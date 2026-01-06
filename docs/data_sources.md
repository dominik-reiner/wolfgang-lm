# Data Sources Documentation

## Overview
This document details the datasets acquired and used for the WOLFGANG-LM project. The data is divided into two primary categories: **Foundation Data** (General Language) and **Persona Data** (Fine-tuning).

## 1. Foundation Dataset (`data/Belletristik`)
*   **Purpose**: Used for the Pre-training phase (Causal Language Modeling).
*   **Content**: A curated collection of German literature (Belletristik) from the period 1600–1900.
*   **Source**: **DTA Normalized Corpus (2020-10-23)**. Downloaded directly via `python -m wolfgang_lm.data.setup`.
*   **Size (Cleaned)**: ~210 MB
*   **File Count**: 548 Text Files
*   **Key Authors**: Goethe, Schiller, Lessing, Kleist, Novalis, Fontane, and others.
*   **Characteristics**:
    *   High-fidelity literary German.
    *   Includes archaic spellings and typography.
    *   Rich in philosophical, poetic, and narrative structures.

## 2. Persona Dataset (`data/gespraeche`)
*   **Purpose**: Used for the Fine-tuning phase (Supervised Instruction Tuning).
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
    *   **Processing**: See [Fine-Tuning Preparation](fine_tuning_preparation.md) for details on how this narrative text is converted into a dialogue format.

## 3. Data Lineage & Changes
*   **Original Location**: All files were originally bulk-downloaded into `data/Belletristik`.
*   **Restructuring**: Identified Eckermann files within `Belletristik` and moved them to a dedicated `data/gespraeche` directory to ensure clean separation between pre-training and fine-tuning data.

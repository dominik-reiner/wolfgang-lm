# Data Preprocessing Documentation

## Overview
This document details the end-to-end pipeline used to transform raw historical texts into the binary format required for training **Wolfgang-LM**. The pipeline consists of five distinct stages, executed sequentially.

## 1. Setup & Download
*   **Script**: `wolfgang_lm/data/setup.py`
*   **Command**: `pixi run python -m wolfgang_lm.data.setup`
*   **Action**: 
    - Downloads the **DTA Normalized Corpus (2020-10-23)**.
    - Extracts zip archives.
    - Organizes files into categorized directories (e.g., `data/Belletristik_Core`, `data/Wissenschaft`).
    - Separates Eckermann's conversations into `data/gespraeche`.

## 2. Cleaning & Normalization
*   **Script**: `wolfgang_lm/data/clean.py`
*   **Command**: `pixi run python -m wolfgang_lm.data.clean`
*   **Action**:
    - **Artifact Removal**: Strips OCR artifacts, page numbers (e.g., `[0012]`), and form feeds (`\x0c`).
    - **Normalization**: Relies on the DTA's pre-normalized text (mapping archaic `ſ` -> `s`, `uͤ` -> `ü`) but ensures clean line breaks.
    - **Output**: Saves processed text files to `data_clean/`.

## 3. Flattening (Corpus Construction)
*   **Script**: `wolfgang_lm/data/flatten.py`
*   **Command**: `pixi run python -m wolfgang_lm.data.flatten`
*   **Action**:
    - Reads all text files from `data_clean/` subdirectories.
    - Concatenates them into a single massive text file: `data_clean/corpus_pretrain.txt`.
    - Inserts a custom **EOS Token** (`<|endoftext|>`) between separate documents to prevent context bleeding during training.

## 4. Tokenization
*   **Script**: `wolfgang_lm/data/tokenizer/train_tokenizer.py`
*   **Command**: `pixi run python -m wolfgang_lm.data.tokenizer.train_tokenizer`
*   **Action**:
    - Trains a **Byte-Level BPE Tokenizer** on `data_clean/corpus_pretrain.txt`.
    - **Vocab Size**: 32,768.
    - **Special Tokens**: `<|endoftext|>`, `<|padding|>`, `<|system|>`, `<|user|>`, `<|assistant|>`.
    - **Output**: Saves the tokenizer definition to `data_clean/tokenizer.json`.

## 5. Binary Conversion
*   **Script**: `wolfgang_lm/data/prepare.py`
*   **Command**: `pixi run python -m wolfgang_lm.data.prepare`
*   **Action**:
    - Reads the flattened corpus `data_clean/corpus_pretrain.txt`.
    - Encodes the text using the trained tokenizer.
    - Saves the compressed token IDs as a memory-mapped binary file: `data_clean/train.bin` (uint16 array).
    - **Result**: This binary file is the direct input for the training loop.


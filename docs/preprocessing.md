# Data Preprocessing Documentation

## Overview
This document explains the cleaning and normalization pipeline implemented in `wolfgang_lm.data.clean`. The goal of this pipeline is to prepare the raw historical text for efficient tokenization and modeling while preserving its semantic value.

## 1. Cleaning Strategy (Artifact Removal)
Raw text from DTA can potentially contain OCR artifacts, page numbers, and editorial markers that must be removed to prevent the model from learning them as part of the language.

### Rules Implemented
1.  **Page Numbers**: 
    - Lines matching patterns like `[0012]`, `[[1]/0011]` are filtered out.
    - Isolated numbers on a single line are removed.
2.  **Form Feeds**:
    - The form feed character `\x0c` (used to mark new pages in older printers) is stripped.
3.  **Heuristics**:
    - Lines that are empty or contain only whitespace are preserved (as single newlines) to maintain paragraph structure but reduce noise.

## 2. Normalization Strategy (Linguistic Standardization)
*   **Approach**: We utilize the **DTA Normalized Version**.
*   **Reasoning**: The DTA team provides expertly curated normalized versions where historical characters like `ſ` (long s), `uͤ` (ü), etc., are already mapped to modern equivalents.
*   **Implementation**: `wolfgang_lm.data.setup` downloads this specific normalized version directly. `wolfgang_lm.data.clean` therefore acts as a pass-through for normalization, focusing only on artifact removal.

## 3. Execution
*   **Script**: `python -m wolfgang_lm.data.clean`
*   **Input**: `data/` (Raw files)
*   **Output**: `data_clean/` (Processed files)

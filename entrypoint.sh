#!/bin/bash
# Entrypoint script for Hugging Face Spaces

MODEL_DIR="out-finetune"
CKPT_FILE="$MODEL_DIR/finetune_ckpt.pt"

# HF_MODEL_ID should be set as a Space secret or environment variable
# Example: "your-username/wolfgang-lm-weights"
HF_MODEL_ID="${HF_MODEL_ID:-dominik-reiner/wolfgang-lm-75M-instruct-v1}"
HF_FILENAME="original/finetune_ckpt.pt"

# Create model directory
mkdir -p "$MODEL_DIR"

# Download weights from Hugging Face Hub if not present
if [ ! -f "$CKPT_FILE" ]; then
    echo "Downloading model weights from Hugging Face Hub..."
    # Dependencies are already installed via pixi (see pyproject.toml)
    # 1. Download Model Checkpoint
    echo "Downloading checkpoint..."
    HF_MODEL_ID="$HF_MODEL_ID" HF_FILENAME="original/finetune_ckpt.pt" MODEL_DIR="$MODEL_DIR" pixi run python wolfgang_lm/utils/download_model.py
    
    # Move checkpoint
    if [ -f "$MODEL_DIR/original/finetune_ckpt.pt" ]; then
        mv "$MODEL_DIR/original/finetune_ckpt.pt" "$CKPT_FILE"
        rm -rf "$MODEL_DIR/original"
    fi

    # 2. Download Tokenizer and move to data_clean/
    echo "Downloading tokenizer..."
    # Ensure data_clean exists
    mkdir -p data_clean
    
    # Download and move tokenizer.json
    HF_MODEL_ID="$HF_MODEL_ID" HF_FILENAME="original/tokenizer.json" MODEL_DIR="$MODEL_DIR" pixi run python wolfgang_lm/utils/download_model.py
    if [ -f "$MODEL_DIR/original/tokenizer.json" ]; then
        mv "$MODEL_DIR/original/tokenizer.json" "data_clean/tokenizer.json"
    fi
    
    
    # Clean up original folder only if empty (it might be empty now if checks moved everything)
    rm -rf "$MODEL_DIR/original"
else
    echo "Model weights already present."
fi

# Start the server (HF Spaces expects port 7860)
echo "Starting Wolfgang-LM server..."
exec pixi run server --host 0.0.0.0 --port 7860

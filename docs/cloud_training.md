# Cloud Training Workflow

This guide details how to train Wolfgang-LM on a remote Cloud GPU instance. We used **RunPod**.

## 1. Local Preparation (Mac)

Before renting a server, verify your code and compile your data locally to save money and upload time.

1.  **Verify Architecture**:
    Run the Verification script to checks for shape errors and parameter counts.
    ```bash
    pixi run python -m wolfgang_lm.modeling.verify
    ```

2.  **Compile Training Data**:
    Convert the raw text corpus into a compact binary format (`train.bin`).
    ```bash
    pixi run python -m wolfgang_lm.data.prepare
    ```
    *Result: `data_clean/train.bin` (~450MB)*

3.  **Package the Project**:
    Create a zip file to easily transfer code + data in one go.
    ```bash
    zip -r wolfgang_lm.zip wolfgang_lm/ data_clean/train.bin data_clean/finetune.bin data_clean/tokenizer.json pyproject.toml pixi.lock -x "*/__pycache__/*" -x "*.pyc"
    ```

## 2. RunPod Setup

### A. Rent Instance
*   **GPU**: NVIDIA RTX 4090 recommended.
*   **Image**: Select **RunPod Pytorch 2.2.0** (Torch 2.2, CUDA 12.1).
    *   *Why?* This image contains the correct GPU drivers (compatible with CUDA 12.1) that match our project settings.
*   **Disk**: 20GB+ Container Disk.

### B. Connect & Install
1.  **Transfer File**:
    Find your SSH command in RunPod (e.g., `ssh root@123.456.78.9 -p 12345`). Use `scp` with the `-P` flag for the port:
    ```bash
    scp -P [PORT] wolfgang_lm.zip root@[IP]:/workspace/
    ```
    ssh root@149.36.1.192 -p 28797 -i ~/.ssh/id_ed25519
    scp -P 28797 wolfgang_lm.zip root@149.36.1.192:/workspace/

2.  **Setup Environment**:
    Connect via SSH or use the Web Terminal.
    ```bash
    cd /workspace/
    
    # 1. Install System Dependencies (Unzip)
    apt-get update && apt-get install -y unzip

    # 2. Install Pixi
    curl -fsSL https://pixi.sh/install.sh | bash
    source ~/.bashrc
    
    # 3. Extract Project
    unzip wolfgang_lm.zip -d wolfgang-lm
    cd wolfgang-lm
    
    # 4. Install Cloud Environment
    # This installs our specific CUDA 12.1 toolkit isolated from the system
    pixi install -e cloud
    ```

## 3. Launch Training

Run the optimized training script. It utilizes **bfloat16** precision, **torch.compile**, and **Fused AdamW**.

```bash
pixi run -e cloud python -m wolfgang_lm.training.train
```

*Expected Duration (RTX 4090): ~45 Minutes for 8000 steps (approx 4 epochs).*
*Note: We train for **4 Epochs** over our 254M token dataset (Total: ~1B tokens).
Calculation: `(254,000,000 tokens * 4 epochs) / (131,072 tokens/step) ≈ 7750 steps`.
This aligns with "Scaling Laws for Data-Constrained Language Model Training" (Muennighoff et al.), ensuring we maximize performance without severe overfitting.*

## 4. Download Pre-Trained Model

Once "Training Complete" appears, download the checkpoint back to your Mac for inference.

```bash
# On your Local Mac (replace PORT and IP)
scp -P [PORT] root@[IP]:/workspace/wolfgang-lm/out-pretrain/ckpt_final.pt ./out-pretrain/
```

## 5. Fine-Tuning

This stage adapts the pre-trained model to a specific conversational style (Johann Wolfgang von Goethe) using your JSONL dataset.

### A. Local Preparation
1.  **Prepare Data**:
    Run the fine-tuning preparation script to generate the masked binary files:
    ```bash
    pixi run python -m wolfgang_lm.data.prepare_finetune
    ```
    *Result: Generates `finetune_dataset.pt` in `data_clean/`.*

2.  **Package Files** (For New Instance):
    If you are setting up a *fresh* instance, create a zip that includes the fine-tuning data AND the pre-trained checkpoint.
    ```bash
    zip -r wolfgang_finetune.zip wolfgang_lm/ out-pretrain/ckpt_final.pt data_clean/finetune_dataset.pt data_clean/tokenizer.json pyproject.toml pixi.lock -x "*/__pycache__/*"
    ```

### B. Upload & Setup
*If you are continuing immediately after Pre-Training (Step 3) on the same instance, skip to **C**.*

If this is a new session:
1.  **Upload Code/Data**:
    Upload and extract your `wolfgang_finetune.zip` as described in Section 2.

### C. Run Fine-Tuning
Execute the fine-tuning script. It is configured to load from `out-pretrain/ckpt_final.pt`.

```bash
pixi run -e cloud python -m wolfgang_lm.training.train_finetune
```

*   **Output**: Checkpoints are saved to `out-finetune/`.
*   **Duration**: Typically much faster than pre-training (e.g., 200 steps).

### D. Download Fine-Tuned Model
Download the final adapted model.

```bash
scp -P [PORT] root@[IP]:/workspace/wolfgang-lm/out-finetune/finetune_ckpt_final.pt ./out-finetune/
```

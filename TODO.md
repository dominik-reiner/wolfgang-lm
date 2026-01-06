# Project Roadmap & TODOs

## missing Control Tokens
- [ ] **Update Tokenizer**: The current tokenizer (`train_tokenizer.py`) only defines `<|endoftext|>` and `<|padding|>`. We need to add conversational control tokens:
  - `<|system|>`
  - `<|user|>`
  - `<|assistant|>`
- [ ] **Regenerate Tokenizer**: Run `train_tokenizer.py` to produce the updated `tokenizer.json` containing these special tokens.

## Data Pipeline for Fine-Tuning
- [ ] **Create `prepare_finetune.py`**: A new script is needed to process `dataset_finetuning.jsonl`.
  - **Logic**: Read JSONL -> Apply Format (`<|user|>...<|assistant|>...`) -> Tokenize -> Save to binary (`finetune.bin`).
  - **Splits**: Create distinct `train` and `val` splits for the fine-tuning phase.

## Fine-Tuning Training Script
- [ ] **Create `train_finetune.py`**: The current `train.py` is designed for pre-training (training from scratch on a raw stream). We need a specialized script for Supervised Fine-Tuning (SFT).
  - **Load Pre-trained Weights**: Must initialize the model from the `out-pretrain/ckpt_final.pt` checkpoint instead of random initialization.
  - **Instruction Masking (Optional but Recommended)**: Modify the loss calculation to only penalize the model for the *Assistant's* response, ignoring the *User's* prompt (loss masking).
  - **Stop Tokens**: Ensure the training considers the end-of-turn tokens correctly.

## Inference & Server
- [ ] **Update `server.py`**:
  - The hardcoded template in `server.py` (L61-L70) must match exactly what `prepare_finetune.py` produces.
- [ ] **Update Generation Logic**:
  - The model should be configured to stop generating when it encounters `<|user|>` or `<|endoftext|>`.

## General
- [ ] **Model config**: Ensure `vocab_size` in `ModelConfig` matches the updated tokenizer size if new tokens push it beyond the limit (though usually we stay within the fixed vocabulary size).

import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from wolfgang_lm.modeling.transformer import WolfgangGPT
from wolfgang_lm.modeling.config import ModelConfig


class WolfgangGenerator:
    def __init__(self, checkpoint_path, tokenizer_path, device=None):
        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = device

        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        if not os.path.exists(checkpoint_path):
            # Basic handling, though caller might want to handle this
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # Convert dictionary config back to ModelConfig object
        config_dict = checkpoint["model_config"]
        # Handle case where checkpoint saves dict vs object
        if isinstance(config_dict, dict):
            self.config = ModelConfig(**config_dict)
        else:
            self.config = config_dict

        self.model = WolfgangGPT(self.config)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_k=200,
        top_p=1.0,
        include_prompt=False,
    ):
        # Encode
        ids = self.tokenizer.encode(prompt).ids
        idx = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

        # Generate
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # crop context
                idx_cond = idx[:, -self.config.block_size :]

                # forward
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature

                # top-k
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = -float("Inf")

                # softmax
                probs = F.softmax(logits, dim=-1)

                # sample
                idx_next = torch.multinomial(probs, num_samples=1)

                # append
                idx = torch.cat((idx, idx_next), dim=1)

        # Decode
        if include_prompt:
            output_text = self.tokenizer.decode(idx[0].tolist())
        else:
            output_text = self.tokenizer.decode(idx[0, len(ids) :].tolist())
        return output_text

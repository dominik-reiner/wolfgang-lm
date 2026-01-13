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

        # Fix state dict keys if model was compiled (remove _orig_mod prefix)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_k=200,
        top_p=1.0,
        repetition_penalty=1.2,
        include_prompt=False,
        stop_tokens=None,
        seed=None,
    ):
        # Encode
        ids = self.tokenizer.encode(prompt).ids
        idx = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

        # Generate
        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            for _ in range(max_new_tokens):
                # crop context
                idx_cond = idx[:, -self.config.block_size :]

                # forward
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature

                # repetition penalty
                # Implements CTRL (Keskar et al., 2019) repetition penalty.
                # If score < 0: multiply by penalty (make more negative).
                # If score > 0: divide by penalty (shrink towards 0).
                if repetition_penalty != 1.0:
                    for i in range(logits.shape[0]):
                        unique_tokens = torch.unique(idx[i])
                        logits[i, unique_tokens] = torch.where(
                            logits[i, unique_tokens] < 0,
                            logits[i, unique_tokens] * repetition_penalty,
                            logits[i, unique_tokens] / repetition_penalty,
                        )

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

                # stop condition
                if stop_tokens is not None and idx_next.item() in stop_tokens:
                    break

                # append
                idx = torch.cat((idx, idx_next), dim=1)

        # Decode
        if include_prompt:
            output_text = self.tokenizer.decode(idx[0].tolist())
        else:
            output_text = self.tokenizer.decode(idx[0, len(ids) :].tolist())

        # Cleanup: Replace sentencepiece underscore if present (Lazy Fix)
        output_text = output_text.replace("\u2581", " ").replace("_", " ")

        return output_text

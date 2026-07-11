"""
GLTR (Giant Language model Test Room) detector.
Gehrmann et al. (2019): "GLTR: Statistical Detection and Visualization of
Generated Text over a Corpus". ACL 2019.

Score = fraction of tokens whose rank in the LM vocabulary distribution is <= top_k.
Higher score → more likely LLM-generated (LLMs tend to use high-probability tokens).
"""

import numpy as np
import torch
from torch import nn
from model import load_model, load_tokenizer


class GLTR(nn.Module):
    def __init__(
        self,
        model_name: str = "gemma-1b",
        device: str = "cuda",
        cache_dir: str = "../cache",
        top_k: int = 10,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.device = device
        self.top_k = top_k
        self.max_length = max_length
        self.scoring_tokenizer = load_tokenizer(model_name, cache_dir)
        self.scoring_model = load_model(model_name, device=device, cache_dir=cache_dir)
        self.scoring_model.eval()

    @torch.inference_mode()
    def forward(self, text: str) -> float:
        """
        Return fraction of tokens ranked <= top_k in the LM distribution.
        Higher → more likely machine-generated.
        """
        tokens = self.scoring_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        ).to(self.device)

        input_ids = tokens["input_ids"]
        T = input_ids.shape[1]
        if T < 2:
            return 0.0

        logits = self.scoring_model(**tokens).logits  # (1, T, V)

        # Predict token t from tokens 0..t-1
        pred_logits = logits[0, :-1, :]   # (T-1, V)
        target_ids = input_ids[0, 1:]     # (T-1,)

        # Rank of each target token in descending-probability order (1-indexed)
        # argsort(argsort(x, descending=True)) gives rank of each vocab element
        ranks_all = torch.argsort(
            torch.argsort(pred_logits, dim=-1, descending=True), dim=-1
        )  # (T-1, V)
        token_ranks = ranks_all[torch.arange(len(target_ids), device=self.device), target_ids] + 1

        fraction = float((token_ranks <= self.top_k).float().mean().item())
        return fraction

    def __call__(self, text: str) -> float:
        return self.forward(text)

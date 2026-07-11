"""
Likelihood detector: mean log-probability of target tokens under a language model.
Based on the log-probability criterion from GLTR (Gehrmann et al., ACL 2019).
Higher score → more likely LLM-generated.
"""
import torch
import torch.nn.functional as F
from model import load_model, load_tokenizer


class Likelihood(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "gemma-1b",
        device: str = "cuda",
        cache_dir: str = "../cache",
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.scoring_tokenizer = load_tokenizer(model_name, cache_dir)
        self.scoring_model = load_model(model_name, device=device, cache_dir=cache_dir)
        self.scoring_model.eval()

    @torch.inference_mode()
    def forward(self, text: str) -> float:
        tokens = self.scoring_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        ).to(self.device)
        input_ids = tokens["input_ids"]
        if input_ids.shape[1] < 2:
            return 0.0
        logits = self.scoring_model(**tokens).logits[:, :-1]  # (1, T-1, V)
        labels = input_ids[:, 1:]                             # (1, T-1)
        log_probs = F.log_softmax(logits, dim=-1)
        log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).mean()
        return log_likelihood.item()

    def __call__(self, text: str) -> float:
        return self.forward(text)

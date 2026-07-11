# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
from model import load_tokenizer, load_model
from scipy.stats import norm


# Considering balanced classification that p(D0) equals to p(D1), we have
#   p(D1|x) = p(x|D1) / (p(x|D1) + p(x|D0))
def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob

class NFastDetectGPT(nn.Module):
    def __init__(self,
                 scoring_model_name: str = "falcon-7b-instruct",
                 sampling_model_name: str = "falcon-7b",
                 device: str = "cuda",
                 cache_dir: str = "../cache",
                 ) -> None:
        super().__init__()
        self.device = device
        self.sampling_model_name = sampling_model_name
        self.scoring_model_name = scoring_model_name
        self.scoring_tokenizer = load_tokenizer(scoring_model_name, cache_dir)
        self.scoring_model = load_model(scoring_model_name, device, cache_dir)
        self.scoring_model.eval()
        if sampling_model_name != scoring_model_name:
            self.sampling_tokenizer = load_tokenizer(sampling_model_name, cache_dir)
            self.sampling_model = load_model(sampling_model_name, device, cache_dir)
            self.sampling_model.eval()

    # compute unnormalized discrepancy and variance (analogous to normalized_model.py predict())
    def forward(self, text: str):
        tokenized = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.sampling_model_name == self.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.sampling_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model(**tokenized).logits[:, :-1]

            if logits_ref.size(-1) != logits_score.size(-1):
                vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
                logits_ref = logits_ref[:, :, :vocab_size]
                logits_score = logits_score[:, :, :vocab_size]

            labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
            lprobs_score = torch.log_softmax(logits_score, dim=-1)
            probs_ref = torch.softmax(logits_ref, dim=-1)

            log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
            mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
            var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)

            # unnormalized discrepancy (mean over tokens, NOT divided by sqrt(var))
            discrepancy = log_likelihood.mean(dim=-1) - mean_ref.mean(dim=-1)
            var_ref = var_ref.sum(dim=-1).clamp_min(0.0001)

        return discrepancy, var_ref


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_model_name', type=str, default="falcon-7b")
    parser.add_argument('--scoring_model_name', type=str, default="falcon-7b-instruct")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    nfdgpt = NFastDetectGPT(args.scoring_model_name, args.sampling_model_name, args.device, args.cache_dir)
    nfdgpt("Warm greetings to you!")

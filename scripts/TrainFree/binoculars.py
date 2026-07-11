from typing import Union
import numpy as np
from torch import nn
import torch
import transformers
from transformers import AutoTokenizer
from model import load_tokenizer, load_model

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

torch.set_grad_enabled(False)

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl


def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce


def assert_tokenizer_consistency(model_id_1, model_id_2, cache_dir):
    identical_tokenizers = (
            AutoTokenizer.from_pretrained(model_id_1, cache_dir=cache_dir).vocab
            == AutoTokenizer.from_pretrained(model_id_2, cache_dir=cache_dir).vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")


class Binoculars(nn.Module):
    def __init__(self,
                 observer_name_or_path: str = "falcon-7b",
                 performer_name_or_path: str = "falcon-7b-instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 cache_dir: str = "../cache",
                 device: str = 'cuda', 
                 ) -> None:
        super().__init__()
        self.change_mode(mode)
        self.observer_model = load_model(observer_name_or_path, device=device, cache_dir=cache_dir, torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32)
        self.performer_model = load_model(performer_name_or_path, device=device, cache_dir=cache_dir, torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32)

        self.observer_model.eval()
        self.performer_model.eval()

        self.scoring_tokenizer = load_tokenizer(observer_name_or_path, cache_dir)
        if not self.scoring_tokenizer.pad_token:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.scoring_tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def forward(self, input_text: str) -> float:
        batch = [input_text]
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.scoring_tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        if np.isnan(binoculars_scores)[0]:
            binoculars_scores = np.array([-1e5], dtype=np.float32)
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0]

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = np.array(self.forward(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model1_name', type=str, default="tiiuae/falcon-7b")
    parser.add_argument('--model2_name', type=str, default="tiiuae/falcon-7b-instruct")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    bino = Binoculars(args.model1_name, args.model2_name)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    bino.compute_score("Warm greetings to you!")
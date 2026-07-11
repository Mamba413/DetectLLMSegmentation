from FineTune.model import ComputeStat
from FineTune.normalized_model import ComputeStat as NormalizedComputeStat
from FineTune.engine import set_seed
from TrainFree.binoculars import Binoculars
from TrainFree.fdgpt import FastDetectGPT
from TrainFree.normalized_fdgpt import NFastDetectGPT
from TrainFree.gltr import GLTR
from TrainFree.likelihood import Likelihood
from TrainFree.lrr import LRR
from device_utils import resolve_device
import re
import torch

class LLMDetector:
    def __init__(self, args):
        args.device = resolve_device(args.device)
        if args.phi == "FT":
            self.model = ComputeStat.from_pretrained(args.from_pretrained, args.base_model, args.base_model, device=args.device, cache_dir=args.cache_dir)
            self.model.set_criterion_fn('mean')
        elif args.phi == "NFT":
            self.model = NormalizedComputeStat.from_pretrained(args.from_pretrained, args.base_model, args.base_model, device=args.device, cache_dir=args.cache_dir)
            self.model.set_criterion_fn('mean')
        elif args.phi == "Bino":
            self.model = Binoculars(args.base_model, args.aux_model, cache_dir=args.cache_dir, device=args.device)
        elif args.phi == "FDGPT":
            self.model = FastDetectGPT(args.base_model, args.aux_model, args.device, args.cache_dir)
        elif args.phi == "NFDGPT":
            self.model = NFastDetectGPT(args.base_model, args.aux_model, args.device, args.cache_dir)
        elif args.phi == "GLTR":
            self.model = GLTR(args.base_model, device=args.device, cache_dir=args.cache_dir)
        elif args.phi == "Likelihood":
            self.model = Likelihood(args.base_model, device=args.device, cache_dir=args.cache_dir)
        elif args.phi == "LRR":
            self.model = LRR(args.base_model, device=args.device, cache_dir=args.cache_dir)
        self.args = args
        set_seed(args.seed)
        self.variance = None

    def score(self, sentence_list):
        if self.args.phi == "FT":
            eval_data = [sentence_list, "placeholder"]
            score = self.model(eval_data, training_module=False)['crit'][0]
            return score.detach().cpu().item()
        elif self.args.phi == "NFT":
            eval_data = sentence_list
            score, variance = self.model.predict(eval_data)
            self.variance = variance.detach().cpu().item()
            return score.detach().cpu().item()
        elif self.args.phi == "Bino":
            return self.model(sentence_list)
        elif self.args.phi == "FDGPT":
            return self.model(sentence_list)
        elif self.args.phi == "NFDGPT":
            score, variance = self.model(sentence_list)
            self.variance = variance.detach().cpu().item()
            return score.detach().cpu().item()
        elif self.args.phi == "GLTR":
            return self.model(sentence_list)
        elif self.args.phi == "Likelihood":
            return self.model(sentence_list)
        elif self.args.phi == "LRR":
            return self.model(sentence_list)
        else:
            raise ValueError(f"Unknown detector type: {self.args.phi}")

def llm_inquiry_detector(sens_list, model, tokenizer, model_name, args):
    """
    Returns:
      predict_label_str: str, e.g., "0100110" (length == num_sentence)
    """

    prompt = (
        "You are an expert in determining the LLM-written sentences in text. "
        "Return ONLY the sentence indices (start from 0) that are written by language model.\n\n"
        "Text to be detected is: \"{}\"\n"
        "The sentence indices is:\n"
    )

    # ---------
    # model family dispatch (minimal additions)
    # ---------
    is_gemma = ("gemma-2" in model_name.lower())
    is_qwen = ("qwen" in model_name.lower())

    if is_gemma:
        def _build_inputs(text: str):
            user_prompt = prompt.format(text)
            messages = [{"role": "user", "content": user_prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
            target_device = next(model.parameters()).device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]
            return inputs, prompt_len

    else:
        def _maybe_apply_chat_template(user_prompt: str) -> str:
            # Qwen3：沿用你能跑通的 chat template，但建议关掉 thinking（否则输出会夹杂大量数字/内容）
            if is_qwen and hasattr(tokenizer, "apply_chat_template"):
                try:
                    return tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    return tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            return user_prompt

        def _build_inputs(text: str):
            user_prompt = prompt.format(text)
            full_prompt = _maybe_apply_chat_template(user_prompt)
            inputs = tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).to(args.device)
            prompt_len = inputs["input_ids"].shape[1]
            return inputs, prompt_len

    @torch.inference_mode()
    def _predict_indices(text: str, num_sentence: int) -> set:
        inputs, prompt_len = _build_inputs(text)

        # pad_token 兜底（有些 tokenizer pad 为空）
        pad_id = tokenizer.pad_token_id
        if pad_id is None and tokenizer.eos_token_id is not None:
            pad_id = tokenizer.eos_token_id

        # 生成长度：输出下标列表不可能只用 2 tokens
        max_new = max(16, min(128, 4 * max(1, num_sentence)))

        gen_kwargs = {
            "do_sample": False,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": pad_id,
            "min_new_tokens": 1,
            "max_new_tokens": max_new,
        }

        # 保留你原来的可选参数开关（即便 do_sample=False 通常不会起作用，也不破坏你的接口习惯）
        if (args is not None) and getattr(args, "do_top_p", False):
            gen_kwargs["top_p"] = args.top_p
        if (args is not None) and getattr(args, "do_top_k", False):
            gen_kwargs["top_k"] = args.top_k
        if (args is not None) and getattr(args, "do_temperature", False):
            gen_kwargs["temperature"] = args.temperature

        output_ids = model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0, prompt_len:]
        out_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 解析所有整数，并过滤到 [0, num_sentence-1]
        inds = [int(x) for x in re.findall(r"\d+", out_text)]
        return {i for i in inds if 0 <= i < num_sentence}

    num_sentence = len(sens_list)
    idx_set = _predict_indices(" ".join(sens_list), num_sentence)

    predict_label = [1.0 if i in idx_set else 0.0 for i in range(num_sentence)]
    return predict_label

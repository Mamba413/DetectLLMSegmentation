import argparse
import json
import os
import random
import re
from typing import Callable

import numpy as np
import tqdm

import custom_datasets
from utils import count_sentences


ParagraphRewriter = Callable[[str, int], str]


class SourceSampleRejectedError(RuntimeError):
    """Raised when a provider rejects the source content."""


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def fibonacci_numbers(k: int):
    """Return the first k Fibonacci numbers using 1, 1, 2, ..."""
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    numbers = [1, 1]
    while len(numbers) < k:
        numbers.append(numbers[-1] + numbers[-2])
    return numbers


def fibonacci_paragraph_lengths(k: int, num_paragraphs: int):
    """Right-align the first k Fibonacci numbers in a fixed paragraph count."""
    if num_paragraphs < 1:
        raise ValueError(f"num_paragraphs must be >= 1, got {num_paragraphs}")
    if k > num_paragraphs:
        raise ValueError(
            f"k must be <= num_paragraphs, got k={k} and "
            f"num_paragraphs={num_paragraphs}"
        )

    fibonacci = fibonacci_numbers(k)
    return [1] * (num_paragraphs - k) + fibonacci


def build_fibonacci_paragraph_plan(
    k: int,
    num_paragraphs: int,
    start: int = 0,
    reverse: bool = False,
):
    """Return sentence-index boundaries for a fixed number of paragraphs."""
    paragraph_lengths = fibonacci_paragraph_lengths(k, num_paragraphs)
    if reverse:
        paragraph_lengths.reverse()

    plan = []
    for paragraph_length in paragraph_lengths:
        end = start + paragraph_length
        plan.append((start, end))
        start = end
    return plan


def _clean_generated_text(text: str):
    text = re.sub(r"(?i)^\s*here(?:'s| is)\s+[^:：]+[:：]\s*", "", text.strip())
    return text.strip()


def _generate_exact_sentence_count(
    source_text: str,
    target_sentence_count: int,
    generate_text: ParagraphRewriter,
    max_retries: int,
):
    for _ in range(max_retries):
        generated_text = _clean_generated_text(
            generate_text(source_text, target_sentence_count)
        )
        _, generated_sentences = count_sentences(generated_text)
        if len(generated_sentences) >= target_sentence_count:
            return generated_sentences[:target_sentence_count]

    raise RuntimeError(
        f"Could not generate {target_sentence_count} sentences after "
        f"{max_retries} attempts."
    )


def build_fibonacci_paragraph_sample(
    original_text: str,
    k: int,
    rewrite_paragraph: ParagraphRewriter,
    max_retries: int = 3,
    max_total_sentences: int | None = None,
    paragraphs_per_half: int = 10,
    task: str = "continuation",
):
    """Build one mixed-provenance document whose unit is a paragraph.

    The first half contains only human paragraphs and the second half contains
    only LLM-generated paragraphs. For continuation, the LLM generates the
    entire second half from the complete human context before paragraph
    grouping. The human half right-aligns the first k Fibonacci paragraph
    lengths, padding earlier paragraphs with one-sentence paragraphs. The LLM
    half uses those lengths in reverse order. Both halves have the same
    sentence count, producing exactly one change point.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")
    if paragraphs_per_half < 1:
        raise ValueError(
            f"paragraphs_per_half must be >= 1, got {paragraphs_per_half}"
        )
    if task not in {"rewrite", "polish", "continuation"}:
        raise ValueError(f"Unsupported task: {task}")

    _, original_sentences = count_sentences(original_text)
    if max_total_sentences is not None:
        original_sentences = original_sentences[:max_total_sentences]
    half_sentence_count = sum(fibonacci_paragraph_lengths(k, paragraphs_per_half))
    required_sentence_count = (
        half_sentence_count if task == "continuation" else 2 * half_sentence_count
    )
    if len(original_sentences) < required_sentence_count:
        raise ValueError(
            f"Source has {len(original_sentences)} sentences, but "
            f"{required_sentence_count} are required for k={k} and "
            f"paragraphs_per_half={paragraphs_per_half}."
        )

    human_plan = build_fibonacci_paragraph_plan(k, paragraphs_per_half)
    human_sentences = original_sentences[:half_sentence_count]
    sampled_paragraphs = [
        " ".join(human_sentences[start:end])
        for start, end in human_plan
    ]

    if task == "continuation":
        llm_sentences = _generate_exact_sentence_count(
            " ".join(human_sentences),
            half_sentence_count,
            rewrite_paragraph,
            max_retries,
        )
        llm_plan = build_fibonacci_paragraph_plan(
            k,
            paragraphs_per_half,
            reverse=True,
        )
        sampled_paragraphs.extend(
            " ".join(llm_sentences[start:end])
            for start, end in llm_plan
        )
    else:
        llm_plan = build_fibonacci_paragraph_plan(
            k,
            paragraphs_per_half,
            start=half_sentence_count,
            reverse=True,
        )
        for start, end in llm_plan:
            source_sentences = original_sentences[start:end]
            output_sentences = _generate_exact_sentence_count(
                " ".join(source_sentences),
                len(source_sentences),
                rewrite_paragraph,
                max_retries,
            )
            sampled_paragraphs.append(" ".join(output_sentences))

    source_labels = ["H"] * paragraphs_per_half + ["L"] * paragraphs_per_half

    return {
        "original": original_text,
        "sampled": "\n\n".join(sampled_paragraphs),
        "sampled_sentence": sampled_paragraphs,
        "source_label": source_labels,
    }


def generate_fibonacci_paragraph_dataset(
    original_texts,
    k: int,
    rewrite_paragraph: ParagraphRewriter,
    n_samples: int,
    max_retries: int = 3,
    max_total_sentences: int | None = None,
    paragraphs_per_half: int = 10,
    task: str = "continuation",
):
    """Generate up to n_samples valid one-change-point paragraph documents."""
    data = {
        "original": [],
        "sampled": [],
        "sampled_sentence": [],
        "source_label": [],
    }

    progress = tqdm.tqdm(total=n_samples, desc="Accepted samples")
    for sample_idx, original_text in enumerate(original_texts):
        if len(data["sampled"]) >= n_samples:
            break
        try:
            sample = build_fibonacci_paragraph_sample(
                original_text,
                k,
                rewrite_paragraph,
                max_retries=max_retries,
                max_total_sentences=max_total_sentences,
                paragraphs_per_half=paragraphs_per_half,
                task=task,
            )
        except (RuntimeError, ValueError) as error:
            print(f"Skipping source sample {sample_idx}: {error}")
            continue

        for key in data:
            data[key].append(sample[key])
        progress.update(1)

    progress.close()
    return data


def _is_content_safety_rejection(error: Exception):
    error_text = str(error).lower()
    return any(
        marker in error_text
        for marker in (
            "content violates usage guidelines",
            "safety_check",
            "safety check",
            "content policy",
        )
    )


def _build_rewrite_messages(source_text: str, target_sentence_count: int, task: str):
    if task == "continuation":
        return [
            {
                "role": "system",
                "content": (
                    "Continue the supplied text naturally. Return only the new "
                    "continuation, without repeating the supplied text or adding "
                    "explanations, headings, or commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Continue the following text by writing exactly "
                    f"{target_sentence_count} new sentences. Return the new "
                    f"sentences as one paragraph.\n\n{source_text}"
                ),
            },
        ]

    action = "polish and improve" if task == "polish" else "rewrite"
    system_prompt = (
        "You are an expert English editor. Return only the requested paragraph, "
        "without explanations, headings, or commentary."
    )
    user_prompt = (
        f"{action.capitalize()} the text below as exactly {target_sentence_count} "
        "sentences. Preserve its meaning and key details. Return the sentences "
        "as one paragraph.\n\n"
        f"{source_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class OpenAIParagraphRewriter:
    def __init__(self, args, base_url=None, api_key=None):
        from openai import OpenAI

        self.args = args
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def __call__(self, source_text: str, target_sentence_count: int):
        from openai import PermissionDeniedError

        kwargs = {
            "model": self.args.model_name,
            "messages": _build_rewrite_messages(
                source_text, target_sentence_count, self.args.task
            ),
        }
        if self.args.do_temperature:
            kwargs["temperature"] = self.args.temperature
        if self.args.do_top_p:
            kwargs["top_p"] = self.args.top_p

        try:
            response = self.client.chat.completions.create(**kwargs)
        except PermissionDeniedError as error:
            if not _is_content_safety_rejection(error):
                raise
            raise SourceSampleRejectedError(
                "provider safety filter rejected the source content"
            ) from None
        return response.choices[0].message.content


class ClaudeParagraphRewriter:
    def __init__(self, args):
        from anthropic import Anthropic

        self.args = args
        self.client = Anthropic()

    def __call__(self, source_text: str, target_sentence_count: int):
        messages = _build_rewrite_messages(
            source_text, target_sentence_count, self.args.task
        )
        kwargs = {
            "model": self.args.model_name,
            "max_tokens": self.args.max_new_tokens,
            "system": messages[0]["content"],
            "messages": [messages[1]],
        }
        if self.args.do_temperature:
            kwargs["temperature"] = self.args.temperature
        if self.args.do_top_p:
            kwargs["top_p"] = self.args.top_p

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class GeminiParagraphRewriter:
    def __init__(self, args):
        from google import genai

        self.args = args
        self.client = genai.Client()

    def __call__(self, source_text: str, target_sentence_count: int):
        from google.genai import types

        messages = _build_rewrite_messages(
            source_text, target_sentence_count, self.args.task
        )
        response = self.client.models.generate_content(
            model=self.args.model_name,
            contents=messages[1]["content"],
            config=types.GenerateContentConfig(
                system_instruction=messages[0]["content"],
                temperature=self.args.temperature if self.args.do_temperature else None,
                top_p=self.args.top_p if self.args.do_top_p else None,
                seed=self.args.seed,
            ),
        )
        return response.text


class HuggingFaceParagraphRewriter:
    def __init__(self, args):
        import torch
        from model import load_model, load_tokenizer

        self.args = args
        self.torch = torch
        self.tokenizer = load_tokenizer(args.model_name, args.cache_dir)
        self.model = load_model(args.model_name, args.device, args.cache_dir)
        self.model.eval()

    def __call__(self, source_text: str, target_sentence_count: int):
        messages = _build_rewrite_messages(
            source_text, target_sentence_count, self.args.task
        )
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, ValueError):
            prompt = "\n\n".join(message["content"] for message in messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        generation_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": (
                self.args.do_temperature or self.args.do_top_p or self.args.do_top_k
            ),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.args.do_temperature:
            generation_kwargs["temperature"] = self.args.temperature
        if self.args.do_top_p:
            generation_kwargs["top_p"] = self.args.top_p
        if self.args.do_top_k:
            generation_kwargs["top_k"] = self.args.top_k

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        return self.tokenizer.decode(
            output_ids[0, prompt_length:],
            skip_special_tokens=True,
        )


def create_paragraph_rewriter(args):
    if args.model_name.startswith("gpt-"):
        return OpenAIParagraphRewriter(args)
    if args.model_name.startswith("grok-"):
        return OpenAIParagraphRewriter(
            args,
            base_url="https://api.x.ai/v1",
            api_key=os.environ["XAI_API_KEY"],
        )
    if args.model_name.startswith("claude-"):
        return ClaudeParagraphRewriter(args)
    if args.model_name.startswith("gemini-"):
        return GeminiParagraphRewriter(args)
    return HuggingFaceParagraphRewriter(args)


def load_source_texts(args):
    def strip_newlines(text):
        return " ".join(text.split())

    dataset_keys = {
        "xsum": "document",
        "squad": "context",
        "writing": "document",
        "essay": "document",
        "yelp_polarity": "text",
        "gopalkalpande/bbc-news-summary": "Summaries",
        "ccdv/govreport-summarization": "report",
    }
    dataset = {
        "yelp": "yelp_polarity",
        "bbc": "gopalkalpande/bbc-news-summary",
        "govreport": "ccdv/govreport-summarization",
    }.get(args.dataset, args.dataset)
    dataset_config = {
        "ccdv/govreport-summarization": "document",
    }.get(dataset)

    if dataset in custom_datasets.DATASETS:
        source_texts = custom_datasets.load(dataset, args.cache_dir)
    else:
        key = dataset_keys.get(dataset)
        source_texts = custom_datasets.load_dataset(
            dataset,
            dataset_config,
            split="train",
            cache_dir=args.cache_dir,
        )[key]

    source_texts = list(dict.fromkeys(source_texts))
    return [strip_newlines(text.strip()) for text in source_texts if text.strip()]


def save_data(output_file: str, args, data):
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_file}.args.json", "w") as fout:
        json.dump(vars(args), fout, indent=4)
    with open(f"{output_file}.raw_data.json", "w") as fout:
        json.dump(data, fout, indent=4)


def forward(args):
    if args.n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {args.n_samples}")
    if args.k < 2:
        raise ValueError(f"k must be >= 2, got {args.k}")
    if args.max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {args.max_retries}")
    if args.paragraphs_per_half < 1:
        raise ValueError(
            f"paragraphs_per_half must be >= 1, got {args.paragraphs_per_half}"
        )
    if args.k > args.paragraphs_per_half:
        raise ValueError(
            f"k must be <= paragraphs_per_half, got k={args.k} and "
            f"paragraphs_per_half={args.paragraphs_per_half}"
        )
    if args.max_total_sentences is not None and args.max_total_sentences < 2:
        raise ValueError(
            f"max_total_sentences must be >= 2, got {args.max_total_sentences}"
        )

    original_texts = load_source_texts(args)
    rewrite_paragraph = create_paragraph_rewriter(args)
    data = generate_fibonacci_paragraph_dataset(
        original_texts,
        args.k,
        rewrite_paragraph,
        args.n_samples,
        max_retries=args.max_retries,
        max_total_sentences=args.max_total_sentences,
        paragraphs_per_half=args.paragraphs_per_half,
        task=args.task,
    )
    save_data(args.output_file, args, data)
    print(f"Generated {len(data['sampled'])} paragraph-level samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="./exp_paragraph/data/writing_gpt-4o_k3")
    parser.add_argument("--dataset", type=str, default="writing", choices=["xsum", "squad", "writing", "pubmed", "essay", "yelp", "bbc", "govreport"])
    parser.add_argument("--task", type=str, default="continuation", choices=["rewrite", "polish", "continuation"])
    parser.add_argument("--n_samples", type=int, default=25)
    parser.add_argument("--k", type=int, default=3, help="Number of Fibonacci terms right-aligned within each human paragraph half.")
    parser.add_argument("--paragraphs_per_half", type=int, default=10, help="Number of paragraphs in each provenance half. Total paragraphs are twice this value.")
    parser.add_argument("--base_model_name", "--model_name", dest="model_name", type=str, default="gpt-4o")
    parser.add_argument("--max_total_sentences", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--do_top_k", action="store_true")
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--do_top_p", action="store_true")
    parser.add_argument("--top_p", type=float, default=0.96)
    parser.add_argument("--do_temperature", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="../cache")
    args = parser.parse_args()

    set_seed(args.seed)
    forward(args)

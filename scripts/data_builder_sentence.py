import argparse
import os
import numpy as np
import random
import json
import tqdm
import time
import re
import custom_datasets
from utils import count_sentences

try:
    from model import load_model, load_tokenizer
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    pass

def deduce_longer_segment(num_sentence, original_sentence, n_split):
    num_sentence_split = int(num_sentence / n_split)
    if num_sentence_split > 16:
        original_sentence = original_sentence[:(16 * n_split)]
        num_sentence = len(original_sentence)
    return num_sentence, original_sentence

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def load_data(args, dataset, key):
    # strip newlines from each example; replace one or more newlines with a single space
    def _strip_newlines(text):
        return ' '.join(text.split())

    # load data
    if dataset in custom_datasets.DATASETS:
        data = custom_datasets.load(dataset, args.cache_dir)
    else:
        dataset_config = {
            'ccdv/govreport-summarization': 'document',
        }.get(dataset)
        data = custom_datasets.load_dataset(dataset, dataset_config, split='train', cache_dir=args.cache_dir)[key]

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the base model)
    # then generate n_samples samples

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [_strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    if dataset in ['writing', 'squad', 'xsum', 'yelp_polarity', "essay"]:
        long_data = [x for x in data if len(x.split()) > 800]
        if len(long_data) > 0:
            data = long_data

    data = data[:5_000]

    return data

def llama_filter(messages, generation_args, pipe, output):
    attempt = 0
    while "Can I help you with something else?" in output or output.startswith("I cannot"):
        attempt += 1
        print(f"Can not generate content... Retrying [Attempt {attempt}]")
        output = pipe(messages, **generation_args)
        output = output[0]['generated_text']
        if attempt==15:
            print("Failed to rewrite")
            break
    if output.startswith("Here is a") or output.startswith("Here's a"):
        output = re.sub(r'Here.*?:', '', output, count=1)
    output = output.replace("\n\n","")
    return output


def _split_sentence_indices(num_sentences: int, k: int, split_type: str):
    """Return a list of (start, end) indices that partitions [0, num_sentences) into k segments.

    - equal_len: segments have as equal sentence counts as possible.
    - random: random cut points; segments may be unbalanced.
    """
    if num_sentences < 0:
        raise ValueError(f"num_sentences must be >= 0, got {num_sentences}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if num_sentences == 0:
        return [(0, 0)]

    k = min(k, num_sentences)  # avoid empty segments

    # if split_type == 'random' and k > 1:
    #     # Choose k-1 cut points from 1..num_sentences-1
    #     cut_points = np.random.choice(np.arange(1, num_sentences), size=k - 1, replace=False)
    #     cut_points = sorted(cut_points.tolist())
    #     boundaries = [0] + cut_points + [num_sentences]
    #     return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    # Default: equal_len
    base = num_sentences // k
    remainder = num_sentences % k

    blocks = []
    start = 0
    for i in range(k):
        block_size = base + (1 if i < remainder else 0)
        end = start + block_size
        if block_size > 0:
            blocks.append((start, end))
        start = end
        
    return blocks


def _clean_llm_prefix(text: str) -> str:
    # Remove occasional "Here is/Here's ...:" prefaces
    cleaned = re.sub(r"(?i)^\s*here(?:'s| is)\s+[^:：]+[:：]\s*", "", text.strip())
    return cleaned.strip()


def _get_latest_human_context(sentences, source_labels):
    if not sentences or not source_labels:
        return ""

    end = len(source_labels) - 1
    while end >= 0 and source_labels[end] != "H":
        end -= 1

    if end < 0:
        return ""

    start = end
    while start >= 0 and source_labels[start] == "H":
        start -= 1

    return " ".join(sentences[start + 1:end + 1]).strip()


def _get_summary_target_sentences(args, seg_sentences):
    target = getattr(args, "t", None)
    if target is None:
        return len(seg_sentences)
    return target


def _build_summary_user_prompt(seg_text: str, n_target: int) -> str:
    return (
        f"Please summarize and condense the following text in approximately {n_target} sentences:\n"
        f"{seg_text}"
    )


def _build_local_edit_prompt(task: str, segment_text: str, n_target: int | None = None) -> str:
    if task == "polish":
        return (
            "You are an English writing polish expert. Polish and improve the writing quality of the text below. "
            "Return ONLY the polished version. Do not explain, do not give multiple options, and do not add commentary.\n\n"
            f'Original text: "{segment_text}"\n\n'
            "Here is the polished version:\n"
        )
    if task == "expand":
        return (
            "You are an English writing expert. Expand and elaborate on the text below, adding more detail and context. "
            "Return ONLY the expanded version. Do not explain, do not give multiple options, and do not add commentary.\n\n"
            f'Original text: "{segment_text}"\n\n'
            "Here is the expanded version:\n"
        )
    if task == "summary":
        return (
            "You are an English writing expert. Summarize and condense the text below while preserving the key information. "
            f"Write approximately {n_target} sentences. Return ONLY the summary. Do not explain, do not give multiple options, and do not add commentary.\n\n"
            f'Original text: "{segment_text}"\n\n'
            "Here is the summary:\n"
        )
    return (
        "You are an English rewriting expert and you would rewrite the text without missing the original details. "
        "Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary.\n\n"
        f'Original text: "{segment_text}"\n\n'
        "Here is the rewritten version:\n"
    )


def _cap_sentence_list(original_sentence, n_split, max_total_sentences=None):
    if max_total_sentences is not None:
        return original_sentence[:max_total_sentences]
    num_sentence, trimmed_sentence = deduce_longer_segment(len(original_sentence), original_sentence, n_split)
    return trimmed_sentence[:num_sentence]


def _count_change_points_from_labels(source_labels):
    change_points = 0
    for i in range(1, len(source_labels)):
        if source_labels[i] != source_labels[i - 1]:
            change_points += 1
    return change_points


def _compute_actual_llm_prop(source_labels):
    if not source_labels:
        return 0.0
    llm_count = sum(label == "L" for label in source_labels)
    return llm_count / len(source_labels)


def _make_sample_metadata(source_labels, placement_mode, target_llm_prop=None, llm_block_start=None, llm_block_end=None):
    return {
        "target_llm_prop": target_llm_prop,
        "actual_llm_prop": _compute_actual_llm_prop(source_labels),
        "llm_block_start": llm_block_start,
        "llm_block_end": llm_block_end,
        "placement_mode": placement_mode,
        "true_cp_num": _count_change_points_from_labels(source_labels),
    }


def _get_bigrewrite_system_prompt():
    return (
        "You are an expert English rewriter. Rewrite the text substantially at both lexical and syntactic levels while preserving the original meaning, key details, and discourse role. "
        "Keep the rewritten passage roughly similar in length to the original block. Avoid copying long phrases verbatim unless they are unavoidable names or facts. "
        "Return ONLY the rewritten text. Do not explain, do not give multiple options, and do not add commentary."
    )


def _build_bigrewrite_user_prompt(segment_text: str, n_target: int) -> str:
    return (
        f"Substantially rewrite the following passage in English while preserving its meaning and factual content. "
        f"The original block contains about {n_target} sentences; keep the rewritten passage at a roughly comparable length, but do not force sentence-by-sentence alignment. "
        f"Return ONLY the rewritten passage.\n\n"
        f"{segment_text}"
    )


def _round_machine_sentence_count(num_sentence: int, machine_prop: float) -> int:
    if num_sentence < 2:
        return 0
    machine_count = int(round(machine_prop * num_sentence))
    machine_count = max(1, machine_count)
    machine_count = min(machine_count, num_sentence - 1)
    return machine_count


def _sample_block_start(num_sentence: int, block_len: int, block_position: str) -> int:
    max_start = num_sentence - block_len
    if max_start <= 0:
        return 0
    if block_position == "middle":
        return max_start // 2
    if block_position == "interior" and max_start >= 2:
        return random.randint(1, max_start - 1)
    return random.randint(0, max_start)


def _extract_exact_sentence_count(text: str, n_target: int):
    _, sentences = count_sentences(text)
    if len(sentences) < n_target:
        return None
    return sentences[:n_target]


def _is_reviewer_layout(args):
    return (
        getattr(args, "task", "rewrite") == "bigrewrite"
        or getattr(args, "placement_mode", "alternating") != "alternating"
        or getattr(args, "machine_prop", None) is not None
        or getattr(args, "max_total_sentences", None) is not None
    )

def openai_sampler(original_texts, args):
    from openai import OpenAI
    client = OpenAI()
    n_samples = len(original_texts)

    # kwargs = {"max_tokens": 500}
    kwargs = {"model": args.model_name}
    if args.do_top_p:
        kwargs['top_p'] = args.top_p
    elif args.do_top_k:
        kwargs['top_k'] = args.top_k
    elif args.do_temperature:
        kwargs['temperature'] = 1.0 if args.model_name == 'gpt-5-mini' else args.temperature

    task = getattr(args, 'task', 'rewrite')
    cont_system = (
        "You are a creative writing assistant. Continue the given text naturally and fluently, "
        "matching the style and tone of what came before."
    )
    exsum_expand_system = 'You are a professional writing expert. Expand and elaborate on the given text, adding more detail and context. Return ONLY the expanded version. Do not explain, do not give multiple options, and do not add commentary.'
    exsum_summary_system = 'You are a professional writing expert. Summarize and condense the given text concisely while preserving the key information. Return ONLY the summarized version. Do not explain, do not give multiple options, and do not add commentary.'
    if task == 'polish':
        system_prompt = 'You are a professional polishing expert and you can help polishing this paragraph. Return ONLY the polished version. Do not explain changes, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please polish and improve the writing quality of:'] * n_samples
    elif task == 'expand':
        system_prompt = 'You are a professional writing expert and you can help expanding this paragraph. Return ONLY the expanded version. Do not explain, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please expand and elaborate on:'] * n_samples
    elif task == 'summary':
        system_prompt = exsum_summary_system
        user_prompts = [''] * n_samples
    elif task == 'continuation':
        system_prompt = cont_system
        user_prompts = [''] * n_samples
    elif task == 'exsum':
        system_prompt = None  # determined per segment
        user_prompts = [''] * n_samples
    else:
        system_prompt = 'You are a professional rewriting expert and you can help paraphrasing in English without missing the original details. Please ensure that, the rewritten text has the same number of sentence as the original text.'
        user_prompts = ['Please rewrite:'] * n_samples

    sampled_sentence_list = []
    source_label_list = []
    response_list =[]
    for idx in tqdm.tqdm(range(n_samples)):
        num_sentence, original_sentence = count_sentences(original_texts[idx])
        num_sentence, original_sentence = deduce_longer_segment(num_sentence, original_sentence, args.n_split)
        segments = _split_sentence_indices(num_sentence, args.n_split, args.split_type)

        new_sentences = []
        source_labels = []

        # Alternate segments.
        # non-continuation/exsum tasks (rewrite/polish/expand): L,H,L,H,... starting with L
        # 'continuation' task: H,L,H,L,... starting with H
        # 'exsum' task: random start (H or L), LLM segments use random expand/summary
        llm_first = random.random() < 0.5 if task == 'exsum' else None
        for seg_idx, (start, end) in enumerate(segments):
            seg_sentences = original_sentence[start:end]
            if len(seg_sentences) == 0:
                continue

            if task == 'exsum':
                is_machine = (seg_idx % 2 == 0) if llm_first else (seg_idx % 2 == 1)
            else:
                is_machine = (seg_idx % 2 == 0) if task != 'continuation' else (seg_idx % 2 == 1)
            if not is_machine:
                new_sentences.extend(seg_sentences)
                source_labels.extend(["H"] * len(seg_sentences))
                continue

            if task == 'continuation':
                preceding_text = _get_latest_human_context(new_sentences, source_labels) or " ".join(seg_sentences)
                n_target = len(seg_sentences)
                cont_user = (
                    f"Continue the following text by writing approximately {n_target} more sentences. "
                    f"Return ONLY the new continuation sentences, without repeating the original text.\n\n"
                    f"{preceding_text}"
                )
                print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Continuation) context: {preceding_text[:100]}...")
                messages = [
                    {'role': 'system', 'content': cont_system},
                    {'role': 'user', 'content': cont_user},
                ]
            elif task == 'exsum':
                sub_task = random.choice(['expand', 'summary'])
                seg_text = " ".join(seg_sentences)
                n_target = len(seg_sentences)
                if sub_task == 'expand':
                    seg_system = exsum_expand_system
                    n_expand = n_target + random.randint(3, max(4, n_target // 2))
                    user_content = f'Please expand and elaborate on the following text, writing approximately {n_expand} sentences:\n{seg_text}'
                else:
                    seg_system = exsum_summary_system
                    n_summary = random.randint(3, max(4, n_target // 2))
                    user_content = f'Please summarize and condense the following text in approximately {n_summary} sentences:\n{seg_text}'
                print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (ExSum/{sub_task}) input: {seg_text[:80]}...")
                messages = [
                    {'role': 'system', 'content': seg_system},
                    {'role': 'user', 'content': user_content},
                ]
            elif task == 'summary':
                seg_text = " ".join(seg_sentences)
                n_summary = _get_summary_target_sentences(args, seg_sentences)
                print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Summary) input: {seg_text[:80]}...")
                messages = [
                    {'role': 'system', 'content': exsum_summary_system},
                    {'role': 'user', 'content': _build_summary_user_prompt(seg_text, n_summary)},
                ]
            else:
                sub_original_texts = " ".join(seg_sentences)
                prompt = user_prompts[idx].strip()
                print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Machine) input: {sub_original_texts}")
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f'{prompt}\n{sub_original_texts}'},
                ]
            kwargs["messages"] = messages
            response = client.chat.completions.create(**kwargs)
            output = response.choices[0].message.content
            output = _clean_llm_prefix(output)
            print(f">>> OpenAI response: {output}")

            # ---- 切分 LLM 返回的句子，并对齐句子数 ----
            _, rewritten_sentences = count_sentences(output)

            new_sentences.extend(rewritten_sentences)
            source_labels.extend(["L"] * len(rewritten_sentences))

        sampled_sentence_list.append(new_sentences)
        response_list.append(" ".join(new_sentences))
        source_label_list.append(source_labels)
    return response_list, sampled_sentence_list, source_label_list


def gemma2_sampler(original_texts, args):
    """
    Rewrite alternating segments using google/gemma-2-2b-it.

    Input/output format matches openai_sampler / qwen_sampler:
      returns (response_list, sampled_sentence_list, source_label_list)
    where source labels are "L" (machine) and "H" (human).

    Requirements:
      - count_sentences(text) -> (num_sentence, sentence_list)
      - deduce_longer_segment(num_sentence, sentence_list, args.n_split) -> (num_sentence, sentence_list)
      - _split_sentence_indices(num_sentence, args.n_split, args.split_type) -> list[(start,end)] with python slicing indices
    """
    model_name = "google/gemma-2-2b-it"
    n_samples = len(original_texts)

    cache_dir = getattr(args, "cache_dir", None)
    device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=(torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float16)
        if device.startswith("cuda") else None,
        device_map="auto" if device.startswith("cuda") else None,
    )

    if not device.startswith("cuda"):
        model = model.to(device)

    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    task = getattr(args, 'task', 'rewrite')
    def _count_tokens_plain(text: str) -> int:
        ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
        return max(1, len(ids))

    @torch.inference_mode()
    def _generate_rewrite(segment_text: str, n_target: int | None = None) -> str:
        segment_text_ntokens = _count_tokens_plain(segment_text)
        user_prompt = _build_local_edit_prompt(task, segment_text, n_target)

        # 官方模型卡示例：messages + tokenizer.apply_chat_template(...)。:contentReference[oaicite:3]{index=3}
        messages = [{"role": "user", "content": user_prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )

        # inputs 搬到模型所在设备（device_map="auto" 时用 model.device 不一定准确，稳妥用 inputs 的 device 与 model 的第一参数设备一致）
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "do_sample": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "min_new_tokens": int(0.8 * segment_text_ntokens),
            "max_new_tokens": int(1.2 * segment_text_ntokens),
        }
        if getattr(args, "do_top_p", False):
            gen_kwargs["top_p"] = args.top_p
        if getattr(args, "do_top_k", False):
            gen_kwargs["top_k"] = args.top_k
        if getattr(args, "do_temperature", False):
            gen_kwargs["temperature"] = args.temperature

        output_ids = model.generate(**inputs, **gen_kwargs)

        # Decode only newly generated tokens
        new_tokens = output_ids[0, prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    @torch.inference_mode()
    def _generate_continuation(preceding_text: str, n_target: int) -> str:
        cont_text = (
            f"Continue the following text by writing approximately {n_target} more sentences. "
            f"Return ONLY the new continuation sentences, without repeating the original text.\n\n{preceding_text}"
        )
        seg_ntokens = max(1, len(tokenizer(preceding_text, add_special_tokens=False).get("input_ids", [1])))
        messages = [{"role": "user", "content": cont_text}]
        inputs = tokenizer(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            return_tensors="pt", padding=True, return_token_type_ids=False
        )
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        gen_kwargs = {
            "do_sample": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "min_new_tokens": int(0.8 * seg_ntokens),
            "max_new_tokens": int(1.2 * seg_ntokens),
        }
        if getattr(args, "do_top_p", False):
            gen_kwargs["top_p"] = args.top_p
        if getattr(args, "do_top_k", False):
            gen_kwargs["top_k"] = args.top_k
        if getattr(args, "do_temperature", False):
            gen_kwargs["temperature"] = args.temperature
        output_ids = model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0, prompt_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    sampled_sentence_list = []
    source_label_list = []
    response_list = []

    for idx in tqdm.tqdm(range(n_samples)):
        num_sentence, original_sentence = count_sentences(original_texts[idx])
        num_sentence, original_sentence = deduce_longer_segment(num_sentence, original_sentence, args.n_split)
        segments = _split_sentence_indices(num_sentence, args.n_split, args.split_type)

        new_sentences = []
        source_labels = []

        for seg_idx, (start, end) in enumerate(segments):
            seg_sentences = original_sentence[start:end]
            if len(seg_sentences) == 0:
                continue

            is_machine = (seg_idx % 2 == 0) if task != 'continuation' else (seg_idx % 2 == 1)
            if not is_machine:
                new_sentences.extend(seg_sentences)
                source_labels.extend(["H"] * len(seg_sentences))
                continue

            if task == 'continuation':
                preceding_text = _get_latest_human_context(new_sentences, source_labels) or " ".join(seg_sentences)
                print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Continuation/Gemma2) context: {preceding_text[:100]}...")
                output = _generate_continuation(preceding_text, len(seg_sentences))
            else:
                segment_text = " ".join(seg_sentences)
                print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Machine/Gemma2) input: {segment_text}")
                output = _generate_rewrite(
                    segment_text,
                    _get_summary_target_sentences(args, seg_sentences) if task == 'summary' else None,
                )
            print(f">>> Gemma2 response: {output}")

            _, rewritten_sentences = count_sentences(output)

            new_sentences.extend(rewritten_sentences)
            source_labels.extend(["L"] * len(rewritten_sentences))

        response_list.append(" ".join(new_sentences))
        sampled_sentence_list.append(new_sentences)
        source_label_list.append(source_labels)

    return response_list, sampled_sentence_list, source_label_list

def claude_sample(original_texts, task, args) -> str:
    def _clean_claude_generated_text(text: str) -> str:
        """
        移除类似 "Here's xxx:" 这种前缀提示语，只保留正文
        """
        # 匹配 "Here's ..." 或 "Here is ..." 后跟冒号的部分
        cleaned = re.sub(r"(?i)\bhere(?:'s| is)\s+[^:：]+[:：]\s*", "", text)
        return cleaned.strip()

    from anthropic import Anthropic
    client = Anthropic()
    model_aliases = {
        'claude-3-5-haiku': 'claude-haiku-4-5',
        'claude-3-5-haiku-20241022': 'claude-haiku-4-5',
        'claude-haiku-4-5': 'claude-haiku-4-5',
        'claude-haiku-4-5-20251001': 'claude-haiku-4-5',
    }
    model_full_name = model_aliases.get(args.model_name, args.model_name)
    n_samples = len(original_texts)
    cont_system = (
        "You are a creative writing assistant. Continue the given text naturally and fluently, "
        "matching the style and tone of what came before."
    )
    exsum_expand_system = 'You are a professional writing expert. Expand and elaborate on the given text, adding more detail and context. Return ONLY the expanded version. Do not explain, do not give multiple options, and do not add commentary.'
    exsum_summary_system = 'You are a professional writing expert. Summarize and condense the given text concisely while preserving the key information. Return ONLY the summarized version. Do not explain, do not give multiple options, and do not add commentary.'

    if task == "rewrite":
        system_prompt = 'You are a professional rewriting expert and you can help paraphrasing this paragraph in English without missing the original details. Please keep the length of the rewritten text similar to the original text. Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please rewrite:'] * n_samples
    elif task == "polish":
        system_prompt = 'You are a professional polishing expert and you can help polishing this paragraph. Return ONLY the polished version. Do not explain changes, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please polish and improve the writing quality of:'] * n_samples
    elif task == "expand":
        system_prompt = 'You are a professional writing expert and you can help expanding this paragraph. Return ONLY the expanded version. Do not explain, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please expand and elaborate on:'] * n_samples
    elif task == "summary":
        system_prompt = exsum_summary_system
        user_prompts = [''] * n_samples
    elif task == "continuation":
        system_prompt = cont_system
        user_prompts = [''] * n_samples
    elif task == "exsum":
        system_prompt = None  # determined per segment
        user_prompts = [''] * n_samples
    else:
        system_prompt = 'You are a professional rewriting expert and you can help paraphrasing this paragraph in English without missing the original details. Please keep the length of the rewritten text similar to the original text. Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please rewrite:'] * n_samples
    base_req = {
        "temperature": args.temperature if args.do_temperature else None,
        "top_p": args.top_p if args.do_top_p else None,
        "top_k": args.top_k if args.do_top_k else None,
    }

    retries = 10
    response_list = []
    sampled_sentence_list = []
    source_label_list = []
    for idx in tqdm.tqdm(range(n_samples)):
        num_sentence, original_sentence = count_sentences(original_texts[idx])
        num_sentence, original_sentence = deduce_longer_segment(num_sentence, original_sentence, args.n_split)
        segments = _split_sentence_indices(num_sentence, args.n_split, args.split_type)

        new_sentences = []
        source_labels = []

        llm_first = random.random() < 0.5 if task == 'exsum' else None
        for seg_idx, (start, end) in enumerate(segments):
            seg_sentences = original_sentence[start:end]
            if len(seg_sentences) == 0:
                continue

            if task == 'exsum':
                is_machine = (seg_idx % 2 == 0) if llm_first else (seg_idx % 2 == 1)
            else:
                is_machine = (seg_idx % 2 == 0) if task != 'continuation' else (seg_idx % 2 == 1)
            if not is_machine:
                new_sentences.extend(seg_sentences)
                source_labels.extend(["H"] * len(seg_sentences))
                continue

            if task == 'continuation':
                preceding_text = _get_latest_human_context(new_sentences, source_labels) or " ".join(seg_sentences)
                n_target = len(seg_sentences)
                user_content = (
                    f"Continue the following text by writing approximately {n_target} more sentences. "
                    f"Return ONLY the new continuation sentences, without repeating the original text.\n\n"
                    f"{preceding_text}"
                )
                seg_system = cont_system
                print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Continuation/Claude) context: {preceding_text[:100]}...")
            elif task == 'exsum':
                sub_task = random.choice(['expand', 'summary'])
                seg_text = " ".join(seg_sentences)
                n_target = len(seg_sentences)
                if sub_task == 'expand':
                    seg_system = exsum_expand_system
                    n_expand = n_target + random.randint(3, max(4, n_target // 2))
                    user_content = f'Please expand and elaborate on the following text, writing approximately {n_expand} sentences:\n{seg_text}'
                else:
                    seg_system = exsum_summary_system
                    n_summary = random.randint(3, max(4, n_target // 2))
                    user_content = f'Please summarize and condense the following text in approximately {n_summary} sentences:\n{seg_text}'
                print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (ExSum/{sub_task}/Claude) input: {seg_text[:80]}...")
            elif task == 'summary':
                seg_text = " ".join(seg_sentences)
                n_summary = _get_summary_target_sentences(args, seg_sentences)
                user_content = _build_summary_user_prompt(seg_text, n_summary)
                seg_system = exsum_summary_system
                print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Summary/Claude) input: {seg_text[:80]}...")
            else:
                segment_text = " ".join(seg_sentences)
                prompt = user_prompts[idx].strip()
                user_content = f'{prompt} {segment_text}'.strip()
                seg_system = system_prompt
                print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Machine/Claude) input: {segment_text}")

            req = {"system": seg_system, **base_req}
            response = None
            for i in range(retries):
                try:
                    response = client.messages.create(
                        model=model_full_name,
                        max_tokens=1000,
                        messages=[{"role": "user", "content": user_content}],
                        **{k: v for k, v in req.items() if v is not None}
                    )
                    break
                except Exception as e:
                    wait_time = (2 ** i) + random.uniform(0, 1)
                    print(f"Request failed ({e}), retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)

            if response is None:
                raise RuntimeError(f"Failed after {retries} retries for sample {idx}, segment {seg_idx}")

            output = _clean_claude_generated_text(response.content[0].text.strip())
            print(f">>> Claude response: {output}")

            _, rewritten_sentences = count_sentences(output)
            new_sentences.extend(rewritten_sentences)
            source_labels.extend(["L"] * len(rewritten_sentences))

        response_list.append(" ".join(new_sentences))
        sampled_sentence_list.append(new_sentences)
        source_label_list.append(source_labels)

    return response_list, sampled_sentence_list, source_label_list

def gemini_sample(original_texts, task, args) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client()
    n_samples = len(original_texts)
    cont_system = (
        "You are a creative writing assistant. Continue the given text naturally and fluently, "
        "matching the style and tone of what came before."
    )
    exsum_expand_system = 'You are a professional writing expert. Expand and elaborate on the given text, adding more detail and context. Return ONLY the expanded version. Do not explain, do not give multiple options, and do not add commentary.'
    exsum_summary_system = 'You are a professional writing expert. Summarize and condense the given text concisely while preserving the key information. Return ONLY the summarized version. Do not explain, do not give multiple options, and do not add commentary.'

    if task == "rewrite":
        system_prompt = 'You are a professional rewriting expert and you can help paraphrasing this paragraph in English without missing the original details. Please keep the length of the rewritten text similar to the original text. Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please rewrite:'] * n_samples
    elif task == "polish":
        system_prompt = 'You are a professional polishing expert and you can help polishing this paragraph. Return ONLY the polished version. Do not explain changes, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please polish and improve the writing quality of:'] * n_samples
    elif task == "expand":
        system_prompt = 'You are a professional writing expert and you can help expanding this paragraph. Return ONLY the expanded version. Do not explain, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please expand and elaborate on:'] * n_samples
    elif task == "summary":
        system_prompt = exsum_summary_system
        user_prompts = [''] * n_samples
    elif task == "continuation":
        system_prompt = cont_system
        user_prompts = [''] * n_samples
    elif task == "exsum":
        system_prompt = None
        user_prompts = [''] * n_samples
    else:
        system_prompt = 'You are a professional rewriting expert and you can help paraphrasing this paragraph in English without missing the original details. Please keep the length of the rewritten text similar to the original text. Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please rewrite:'] * n_samples

    max_retries = 5
    base_delay = 2
    response_list = []
    sampled_sentence_list = []
    source_label_list = []
    for idx in tqdm.tqdm(range(n_samples)):
        num_sentence, original_sentence = count_sentences(original_texts[idx])
        num_sentence, original_sentence = deduce_longer_segment(num_sentence, original_sentence, args.n_split)
        segments = _split_sentence_indices(num_sentence, args.n_split, args.split_type)

        new_sentences = []
        source_labels = []
        llm_first = random.random() < 0.5 if task == 'exsum' else None

        for seg_idx, (start, end) in enumerate(segments):
            seg_sentences = original_sentence[start:end]
            if len(seg_sentences) == 0:
                continue

            if task == 'exsum':
                is_machine = (seg_idx % 2 == 0) if llm_first else (seg_idx % 2 == 1)
            else:
                is_machine = (seg_idx % 2 == 0) if task != 'continuation' else (seg_idx % 2 == 1)
            if not is_machine:
                new_sentences.extend(seg_sentences)
                source_labels.extend(["H"] * len(seg_sentences))
                continue

            if task == 'continuation':
                preceding_text = _get_latest_human_context(new_sentences, source_labels) or " ".join(seg_sentences)
                n_target = len(seg_sentences)
                seg_system = cont_system
                user_content = (
                    f"Continue the following text by writing approximately {n_target} more sentences. "
                    f"Return ONLY the new continuation sentences, without repeating the original text.\n\n"
                    f"{preceding_text}"
                )
                print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Continuation/Gemini) context: {preceding_text[:100]}...")
            elif task == 'exsum':
                sub_task = random.choice(['expand', 'summary'])
                seg_text = " ".join(seg_sentences)
                n_target = len(seg_sentences)
                if sub_task == 'expand':
                    seg_system = exsum_expand_system
                    n_expand = n_target + random.randint(3, max(4, n_target // 2))
                    user_content = f'Please expand and elaborate on the following text, writing approximately {n_expand} sentences:\n{seg_text}'
                else:
                    seg_system = exsum_summary_system
                    n_summary = random.randint(3, max(4, n_target // 2))
                    user_content = f'Please summarize and condense the following text in approximately {n_summary} sentences:\n{seg_text}'
                print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (ExSum/{sub_task}/Gemini) input: {seg_text[:80]}...")
            elif task == 'summary':
                seg_text = " ".join(seg_sentences)
                n_summary = _get_summary_target_sentences(args, seg_sentences)
                seg_system = exsum_summary_system
                user_content = _build_summary_user_prompt(seg_text, n_summary)
                print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Summary/Gemini) input: {seg_text[:80]}...")
            else:
                segment_text = " ".join(seg_sentences)
                prompt = user_prompts[idx].strip()
                seg_system = system_prompt
                user_content = f'{prompt}\n{segment_text}'.strip()
                print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Machine/Gemini) input: {segment_text[:80]}...")

            response = None
            for i in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=args.model_name,
                        contents=user_content,
                        config=types.GenerateContentConfig(
                            top_p=args.top_p if args.do_top_p else None,
                            top_k=args.top_k if args.do_top_k else None,
                            temperature=args.temperature if args.do_temperature else None,
                            seed=args.seed,
                            candidate_count=1,
                            system_instruction=seg_system,
                        ),
                    )
                    break
                except Exception as e:
                    print(f"Error: {e}, retry {i+1}/{max_retries}")
                    time.sleep(base_delay * (2 ** i))
            if response is None:
                raise RuntimeError(f"Failed after {max_retries} retries for sample {idx}, segment {seg_idx}")

            output = response.text.strip()
            print(f">>> Gemini response: {output}")
            _, generated_sentences = count_sentences(output)
            new_sentences.extend(generated_sentences)
            source_labels.extend(["L"] * len(generated_sentences))

        response_list.append(" ".join(new_sentences))
        sampled_sentence_list.append(new_sentences)
        source_label_list.append(source_labels)

    return response_list, sampled_sentence_list, source_label_list

def grok_sampler(original_texts, args):
    """Rewrite or continue alternating segments using xAI Grok (OpenAI-compatible API).

    Mirrors openai_sampler but connects to https://api.x.ai/v1.
    Requires environment variable XAI_API_KEY.
    """
    import os
    from openai import OpenAI
    client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"])
    n_samples = len(original_texts)

    kwargs = {"model": args.model_name}
    if args.do_top_p:
        kwargs['top_p'] = args.top_p
    elif args.do_temperature:
        kwargs['temperature'] = args.temperature

    task = getattr(args, 'task', 'rewrite')
    cont_system = (
        "You are a creative writing assistant. Continue the given text naturally and fluently, "
        "matching the style and tone of what came before."
    )
    exsum_expand_system = 'You are a professional writing expert. Expand and elaborate on the given text, adding more detail and context. Return ONLY the expanded version. Do not explain, do not give multiple options, and do not add commentary.'
    exsum_summary_system = 'You are a professional writing expert. Summarize and condense the given text concisely while preserving the key information. Return ONLY the summarized version. Do not explain, do not give multiple options, and do not add commentary.'
    if task == 'polish':
        system_prompt = 'You are a professional polishing expert and you can help polishing this paragraph. Return ONLY the polished version. Do not explain changes, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please polish and improve the writing quality of:'] * n_samples
    elif task == 'expand':
        system_prompt = 'You are a professional writing expert and you can help expanding this paragraph. Return ONLY the expanded version. Do not explain, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please expand and elaborate on:'] * n_samples
    elif task == 'summary':
        system_prompt = exsum_summary_system
        user_prompts = [''] * n_samples
    elif task == 'continuation':
        system_prompt = cont_system
        user_prompts = [''] * n_samples
    elif task == 'exsum':
        system_prompt = None  # determined per segment
        user_prompts = [''] * n_samples
    else:
        system_prompt = 'You are a professional rewriting expert and you can help paraphrasing in English without missing the original details. Please ensure that, the rewritten text has the same number of sentence as the original text.'
        user_prompts = ['Please rewrite:'] * n_samples

    sampled_sentence_list = []
    source_label_list = []
    response_list = []
    metadata_list = []
    accepted_originals = []
    from openai import PermissionDeniedError as _GrokPermissionDeniedError
    max_generation_retries = 4

    def _chat_once(messages, sample_idx, seg_desc):
        kwargs["messages"] = messages
        try:
            response = client.chat.completions.create(**kwargs)
        except _GrokPermissionDeniedError as e:
            print(f">>> Grok 403 blocked sample {sample_idx} ({seg_desc}): {e} — skipping sample")
            return None, True
        output = response.choices[0].message.content
        output = _clean_llm_prefix(output)
        print(f">>> Grok response: {output}")
        return output, False

    def _generate_exact_sentences(messages, n_target, sample_idx, seg_desc):
        for attempt in range(max_generation_retries):
            output, blocked = _chat_once(messages, sample_idx, seg_desc)
            if blocked:
                return None, True
            rewritten_sentences = _extract_exact_sentence_count(output, n_target)
            if rewritten_sentences is not None:
                return rewritten_sentences, False
            print(
                f">>> Grok sentence-count mismatch for sample {sample_idx} ({seg_desc}), "
                f"retry {attempt + 1}/{max_generation_retries}"
            )
        return None, False

    def _generate_flexible_sentences(messages, sample_idx, seg_desc):
        output, blocked = _chat_once(messages, sample_idx, seg_desc)
        if blocked:
            return None, True
        _, generated_sentences = count_sentences(output)
        if len(generated_sentences) == 0:
            print(f">>> Grok returned no usable sentences for sample {sample_idx} ({seg_desc}) — skipping sample")
            return None, False
        return generated_sentences, False

    reviewer_layout = _is_reviewer_layout(args)
    progress_bar = None
    if reviewer_layout:
        progress_bar = tqdm.tqdm(total=args.n_samples, desc="Accepted samples")
        iterable = enumerate(original_texts)
    else:
        iterable = enumerate(tqdm.tqdm(original_texts, total=n_samples))

    for idx, original_text in iterable:
        if len(response_list) >= args.n_samples:
            break

        _, original_sentence = count_sentences(original_text)
        original_sentence = _cap_sentence_list(
            original_sentence,
            args.n_split,
            max_total_sentences=getattr(args, "max_total_sentences", None),
        )
        num_sentence = len(original_sentence)
        if num_sentence < 2:
            continue

        placement_mode = getattr(args, "placement_mode", "alternating")
        sample_blocked = False
        sample_metadata = None

        if task == "continuation" and placement_mode == "suffix":
            machine_prop = getattr(args, "machine_prop", None)
            if machine_prop is None:
                raise ValueError("--machine_prop is required for continuation with --placement_mode suffix")
            n_target = _round_machine_sentence_count(num_sentence, machine_prop)
            n_human = num_sentence - n_target
            human_sentences = original_sentence[:n_human]
            preceding_text = " ".join(human_sentences)
            messages = [
                {'role': 'system', 'content': cont_system},
                {'role': 'user', 'content': (
                    f"Continue the following text by writing exactly {n_target} new sentences. "
                    f"Return ONLY the new continuation sentences, without repeating the original text.\n\n"
                    f"{preceding_text}"
                )},
            ]
            generated_sentences, sample_blocked = _generate_exact_sentences(
                messages, n_target, idx, f"suffix-continuation/{n_target}"
            )
            if sample_blocked or generated_sentences is None:
                continue
            new_sentences = human_sentences + generated_sentences
            source_labels = (["H"] * n_human) + (["L"] * len(generated_sentences))
            sample_metadata = _make_sample_metadata(
                source_labels,
                placement_mode="suffix",
                target_llm_prop=machine_prop,
                llm_block_start=n_human,
                llm_block_end=n_human + len(generated_sentences),
            )
        elif task == "bigrewrite" or placement_mode == "single_block_random":
            machine_prop = getattr(args, "machine_prop", None)
            if machine_prop is None:
                raise ValueError("--machine_prop is required for bigrewrite / single_block_random layouts")
            block_len = _round_machine_sentence_count(num_sentence, machine_prop)
            block_start = _sample_block_start(
                num_sentence,
                block_len,
                getattr(args, "block_position", "anywhere"),
            )
            block_end = block_start + block_len
            seg_sentences = original_sentence[block_start:block_end]
            seg_text = " ".join(seg_sentences)
            messages = [
                {'role': 'system', 'content': _get_bigrewrite_system_prompt()},
                {'role': 'user', 'content': _build_bigrewrite_user_prompt(seg_text, block_len)},
            ]
            rewritten_sentences, sample_blocked = _generate_flexible_sentences(
                messages, idx, f"bigrewrite/{block_start}:{block_end}"
            )
            if sample_blocked or rewritten_sentences is None:
                continue
            new_sentences = original_sentence[:block_start] + rewritten_sentences + original_sentence[block_end:]
            source_labels = (
                (["H"] * block_start)
                + (["L"] * len(rewritten_sentences))
                + (["H"] * (num_sentence - block_end))
            )
            sample_metadata = _make_sample_metadata(
                source_labels,
                placement_mode="single_block_random",
                target_llm_prop=machine_prop,
                llm_block_start=block_start,
                llm_block_end=block_start + len(rewritten_sentences),
            )
        else:
            segments = _split_sentence_indices(num_sentence, args.n_split, args.split_type)
            new_sentences = []
            source_labels = []
            llm_first = random.random() < 0.5 if task == 'exsum' else None
            for seg_idx, (start, end) in enumerate(segments):
                seg_sentences = original_sentence[start:end]
                if len(seg_sentences) == 0:
                    continue

                if task == 'exsum':
                    is_machine = (seg_idx % 2 == 0) if llm_first else (seg_idx % 2 == 1)
                else:
                    is_machine = (seg_idx % 2 == 0) if task != 'continuation' else (seg_idx % 2 == 1)
                if not is_machine:
                    new_sentences.extend(seg_sentences)
                    source_labels.extend(["H"] * len(seg_sentences))
                    continue

                if task == 'continuation':
                    preceding_text = _get_latest_human_context(new_sentences, source_labels) or " ".join(seg_sentences)
                    n_target = len(seg_sentences)
                    messages = [
                        {'role': 'system', 'content': cont_system},
                        {'role': 'user', 'content': (
                            f"Continue the following text by writing approximately {n_target} more sentences. "
                            f"Return ONLY the new continuation sentences, without repeating the original text.\n\n"
                            f"{preceding_text}"
                        )},
                    ]
                elif task == 'exsum':
                    sub_task = random.choice(['expand', 'summary'])
                    seg_text = " ".join(seg_sentences)
                    n_target = len(seg_sentences)
                    if sub_task == 'expand':
                        seg_system = exsum_expand_system
                        n_expand = n_target + random.randint(3, max(4, n_target // 2))
                        user_content = f'Please expand and elaborate on the following text, writing approximately {n_expand} sentences:\n{seg_text}'
                    else:
                        seg_system = exsum_summary_system
                        n_summary = random.randint(3, max(4, n_target // 2))
                        user_content = f'Please summarize and condense the following text in approximately {n_summary} sentences:\n{seg_text}'
                    print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (ExSum/{sub_task}/Grok) input: {seg_text[:80]}...")
                    messages = [
                        {'role': 'system', 'content': seg_system},
                        {'role': 'user', 'content': user_content},
                    ]
                elif task == 'summary':
                    seg_text = " ".join(seg_sentences)
                    n_summary = _get_summary_target_sentences(args, seg_sentences)
                    print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Summary/Grok) input: {seg_text[:80]}...")
                    messages = [
                        {'role': 'system', 'content': exsum_summary_system},
                        {'role': 'user', 'content': _build_summary_user_prompt(seg_text, n_summary)},
                    ]
                else:
                    sub_original_texts = " ".join(seg_sentences)
                    messages = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': f'{user_prompts[idx].strip()}\n{sub_original_texts}'},
                    ]

                output, sample_blocked = _chat_once(messages, idx, f"segment-{seg_idx}")
                if sample_blocked:
                    break
                _, rewritten_sentences = count_sentences(output)
                new_sentences.extend(rewritten_sentences)
                source_labels.extend(["L"] * len(rewritten_sentences))

            if sample_blocked:
                continue
            sample_metadata = _make_sample_metadata(
                source_labels,
                placement_mode=placement_mode,
            )

        sampled_sentence_list.append(new_sentences)
        response_list.append(" ".join(new_sentences))
        source_label_list.append(source_labels)
        metadata_list.append(sample_metadata)
        accepted_originals.append(original_text)
        if progress_bar is not None:
            progress_bar.update(1)

    if len(response_list) < args.n_samples:
        print(f"Collected {len(response_list)} valid samples out of requested {args.n_samples}.")
    if progress_bar is not None:
        progress_bar.close()

    return response_list, sampled_sentence_list, source_label_list, metadata_list, accepted_originals


def save_data(output_file, args, data):
    # write args to file
    args_file = f"{output_file}.args.json"
    with open(args_file, "w") as fout:
        json.dump(args.__dict__, fout, indent=4)
        print(f"Args written into {args_file}")

    # write the data to a json file in the save folder
    data_file = f"{output_file}.raw_data.json"
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"Raw data written into {data_file}")

def forward(args):
    if args.dataset == 'yelp':
        args.dataset = 'yelp_polarity'
    if args.dataset == "bbc":
        args.dataset = 'gopalkalpande/bbc-news-summary'
    if args.dataset == "govreport":
        args.dataset = 'ccdv/govreport-summarization'
    print(f'Loading dataset {args.dataset}...')
    dataset_keys = {
        'xsum': 'document',
        'squad': 'context',
        'writing': 'document',
        'essay': 'document',
        'yelp_polarity': 'text',
        'gopalkalpande/bbc-news-summary': 'Summaries',
        'ccdv/govreport-summarization': 'report',
    }

    original_texts = load_data(args, args.dataset, dataset_keys[args.dataset] if args.dataset in dataset_keys else None)

    # tokenizer = load_tokenizer('gpt-neo-2.7B', cache_dir=args.cache_dir)
    # # keep only examples with <= 2048 tokens according to base_tokenizer
    # # this step has the extra effect of removing examples with low-quality/garbage content
    # tokenized_data = tokenizer(original_texts)
    # original_texts = [x for x, y in zip(original_texts, tokenized_data["input_ids"]) if len(y) <= 2048]

    # print stats about remaining data
    print(f"Total number of samples: {len(original_texts)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in original_texts])}")

    if not (args.n_samples > 0):
        raise ValueError(f"--n_samples must be positive, got {args.n_samples}")

    if _is_reviewer_layout(args) and args.model_name not in ['grok-3', 'grok-3-mini', 'grok-4-1-fast-non-reasoning']:
        raise ValueError("Reviewer-style layouts are only implemented for Grok models.")

    if _is_reviewer_layout(args):
        if args.machine_prop is None:
            raise ValueError("--machine_prop is required for reviewer-style layouts.")
        if not (0.0 < args.machine_prop < 1.0):
            raise ValueError(f"--machine_prop must be in (0, 1), got {args.machine_prop}")
        candidate_texts = original_texts
    else:
        candidate_texts = original_texts[:min(args.n_samples, len(original_texts))]

    # For exsum, use a model-specific seed so different models produce different H/L patterns
    if getattr(args, 'task', 'rewrite') == 'exsum':
        model_seed = args.seed ^ (abs(hash(args.model_name)) % 100000)
        random.seed(model_seed)

    if args.model_name in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o', 'gpt-5', 'gpt-5-mini']:
        sampled_outputs = openai_sampler(candidate_texts, args)
    elif args.model_name in ['grok-3', 'grok-3-mini', 'grok-4-1-fast-non-reasoning']:
        sampled_outputs = grok_sampler(candidate_texts, args)
    elif args.model_name in ['gemma-2b-instruct']:
        sampled_outputs = gemma2_sampler(candidate_texts, args)
    elif args.model_name == 'gemini-2.5-flash':
        sampled_outputs = gemini_sample(candidate_texts, args.task, args)
    elif args.model_name in ['claude-3-5-haiku', 'claude-haiku-4-5']:
        sampled_outputs = claude_sample(candidate_texts, args.task, args)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    sample_metadata_list = None
    accepted_original_texts = None
    if len(sampled_outputs) == 5:
        sampled_texts, sampled_sentence_list, source_label_list, sample_metadata_list, accepted_original_texts = sampled_outputs
    elif len(sampled_outputs) == 4:
        sampled_texts, sampled_sentence_list, source_label_list, sample_metadata_list = sampled_outputs
    else:
        sampled_texts, sampled_sentence_list, source_label_list = sampled_outputs

    if accepted_original_texts is None:
        accepted_original_texts = candidate_texts[:len(sampled_texts)]

    if args.dataset == 'yelp_polarity':
        args.dataset = 'yelp'
    if args.dataset == "gopalkalpande/bbc-news-summary":
        args.dataset = 'bbc'
    if args.dataset == "ccdv/govreport-summarization":
        args.dataset = 'govreport'

    data = {"original": [], "sampled": [], "sampled_sentence": [], "source_label": []}
    if sample_metadata_list is not None:
        data.update({
            "target_llm_prop": [],
            "actual_llm_prop": [],
            "llm_block_start": [],
            "llm_block_end": [],
            "placement_mode": [],
            "true_cp_num": [],
        })

    for i, (o, s, s_sentence, label) in enumerate(zip(accepted_original_texts, sampled_texts, sampled_sentence_list, source_label_list)):
        # add to the data
        data["original"].append(o)
        data["sampled"].append(s)
        data["sampled_sentence"].append(s_sentence)
        data["source_label"].append(label)
        if sample_metadata_list is not None:
            metadata = sample_metadata_list[i]
            data["target_llm_prop"].append(metadata["target_llm_prop"])
            data["actual_llm_prop"].append(metadata["actual_llm_prop"])
            data["llm_block_start"].append(metadata["llm_block_start"])
            data["llm_block_end"].append(metadata["llm_block_end"])
            data["placement_mode"].append(metadata["placement_mode"])
            data["true_cp_num"].append(metadata["true_cp_num"])

    save_data(args.output_file, args, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_location_single_cp/data/squad_gpt-4o_rewrite")
    parser.add_argument('--task', type=str, default="rewrite", choices=["rewrite", "polish", "expand", "generation", "summary", "continuation", "exsum", "bigrewrite"])
    parser.add_argument('--dataset', type=str, default="squad", choices=['xsum', 'squad', 'writing', 'pubmed', 'essay', 'yelp', 'bbc', 'govreport'])
    parser.add_argument('--n_samples', type=int, default=25)
    parser.add_argument(
        '--base_model_name',
        type=str,
        default="gpt-4o",
        choices=["gpt-4o", "gpt-5", "gpt-5-mini", "grok-3", "grok-3-mini", 'grok-4-1-fast-non-reasoning', "gemini-2.5-flash", "claude-3-5-haiku", "claude-haiku-4-5", "gemma-2b-instruct"],
    )
    parser.add_argument('--n_split', type=int, default=2, help='Number of segments K (>=2). Segments alternate Machine/Human.')
    parser.add_argument('--split_type', type=str, default='equal_len', choices=['random', 'equal_len'])
    parser.add_argument('--machine_prop', type=float, default=None, help='Target proportion of LLM-written sentences for reviewer-style layouts.')
    parser.add_argument('--placement_mode', type=str, default='alternating', choices=['alternating', 'suffix', 'single_block_random'])
    parser.add_argument('--block_position', type=str, default='anywhere', choices=['anywhere', 'interior', 'middle'])
    parser.add_argument('--max_total_sentences', type=int, default=None, help='Optional hard cap on the number of source sentences retained before generation.')
    parser.add_argument('--max_new_tokens', type=int, default=1000)
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--do_temperature', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--t', type=int, default=None, help='Optional target sentence count for each machine summary segment when --task summary.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()
    args.model_name = args.base_model_name

    if args.n_split < 2:
        raise ValueError(f"--n_split must be >= 2, got {args.n_split}")
    if args.t is not None and args.t < 1:
        raise ValueError(f"--t must be >= 1, got {args.t}")
    if args.max_total_sentences is not None and args.max_total_sentences < 2:
        raise ValueError(f"--max_total_sentences must be >= 2, got {args.max_total_sentences}")

    set_seed(args.seed)

    forward(args)
    
    

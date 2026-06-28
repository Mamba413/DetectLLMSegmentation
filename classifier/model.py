# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os

def from_pretrained(cls, model_name, kwargs, cache_dir):
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)

# predefined models
model_fullnames = {  'gpt2': 'gpt2',
                     'gpt2-xl': 'gpt2-xl',
                     'opt-2.7b': 'facebook/opt-2.7b',
                     'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
                     'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                     'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
                     'mgpt': 'sberbank-ai/mGPT',
                     'pubmedgpt': 'stanford-crfm/pubmedgpt',
                     'mt5-xl': 'google/mt5-xl',
                     'llama-13b': 'huggyllama/llama-13b',
                     'qwen-7b': 'Qwen/Qwen2.5-7B',
                     'qwen-7b-instruct': 'Qwen/Qwen2.5-7B-Instruct',
                     'mistralai-7b-instruct': 'mistralai/Mistral-7B-Instruct-v0.3',
                     'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
                     'llama3-8b-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
                     'llama4-17b':'meta-llama/Llama-4-Scout-17B-16E-Instruct',
                     'falcon-7b': 'tiiuae/falcon-7b',
                     'llama2-13b': 'TheBloke/Llama-2-13B-fp16',
                     'llama2-13b-chat': 'meta-llama/Llama-2-13b-chat-hf', 
                     'qwen-14b': 'Qwen/Qwen3-14B',
                     'qwen-14b-base': 'Qwen/Qwen3-14B-Base',
                     'gemma-9b': 'google/gemma-2-9b',
                     'gemma-4b': 'google/gemma-3-4b-pt',
                     'gemma-1b': 'google/gemma-3-1b-pt',
                     'gemma-1b-instruct': 'google/gemma-3-1b-it',
                     'gemma-9b-instruct': 'google/gemma-2-9b-it',
                     'gemma-12b': 'google/gemma-3-12b-pt',
                     'gemma-12b-instruct': 'google/gemma-3-12b-it',
                     'bloom-7b1': 'bigscience/bloom-7b1',
                     'opt-13b': 'facebook/opt-13b',
                     'pythia-12b': 'EleutherAI/pythia-12b',
                     'falcon-7b-instruct': 'tiiuae/falcon-7b-instruct', 
                     'mistralai-8b-instruct': 'mistralai/Ministral-8B-Instruct-2410',
                     }
float16_models = ['gpt-neo-2.7B', 'gpt-j-6B', 'gpt-neox-20b', 'qwen-7b', 'qwen-7b-instruct', 'mistralai-7b', 'llama3-8b', 'llama3-8b-instruct', 'llama-13b', 'llama2-13b', 'llama4-17b', 'bloom-7b1', 'opt-13b', 'pythia-12b', 'falcon-7b', 'falcon-7b-instruct', 'llama2-13b-chat', 'qwen-14b', 'qwen-14b-base', 'gemma-12b', 'gemma-12b-instruct', 'gemma-9b', 'gemma-9b-instruct', 'mistralai-7b-base', 'mistralai-7b-instruct', 'gemma-4b', 'mistralai-8b-instruct']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_model(model_name, device, cache_dir, torch_dtype=None):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in float16_models:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in model_name:
        model_kwargs.update(dict(revision='float16'))
    if torch_dtype is not None:
        model_kwargs.update(dict(torch_dtype=torch_dtype))
    model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="mistralai-8b-instruct")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    load_tokenizer(args.model_name, args.cache_dir)
    load_model(args.model_name, 'cpu', args.cache_dir)

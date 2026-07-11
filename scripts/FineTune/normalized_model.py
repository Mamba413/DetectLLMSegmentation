import torch
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json

import os
from device_utils import resolve_device

def calculate_MMD_loss(human_crit, sample_crit):
    mmd_loss = human_crit.mean() - sample_crit.mean()
    return mmd_loss

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    if "/" in model_name:
        local_path = os.path.join(cache_dir, model_name.split("/")[1])
    else:
        local_path = os.path.join(cache_dir, model_name)

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir, device_map='auto')

model_fullnames = {  
    'gemma-1b': 'google/gemma-3-1b-pt',
}
float16_models = []

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_tokenizer(model_name, for_dataset, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    if for_dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    else:
        optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
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
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).clamp_min(0.0001).sqrt()
    
    return discrepancy, log_likelihood.sum(dim=-1)

class ComputeStat(nn.Module):
    def __init__(self, model_name, dataset='xsum', device='cuda', cache_dir='./models'):
        super().__init__()
        self.device = resolve_device(device)
        self.reference_model_name = get_model_fullname(model_name)
        self.scoring_model_name = get_model_fullname(model_name)
        
        def load_model(model_name, device, cache_dir):
            model_fullname = get_model_fullname(model_name)
            print(f'Loading model {model_fullname}...')
            model_kwargs = {}
            if model_name in float16_models:
                model_kwargs.update(dict(torch_dtype=torch.float16))
            if torch.__version__ >= '2.0.0' and 'gemma' in model_name:
                model_kwargs.update({'attn_implementation': 'sdpa'})
            model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
            print(f'Moving model to {device}...', end='', flush=True)
            start = time.time()
            model.to(device)
            print(f'DONE ({time.time() - start:.2f}s)')
            return model
        
        # load scoring model
        self.scoring_tokenizer = load_tokenizer(model_name, dataset, cache_dir)
        scoring_model = load_model(model_name, self.device, cache_dir)
        if model_name in ['gemma-1b']:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=4,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
        else:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=8, 
                lora_alpha=32, 
                lora_dropout=0.1, 
            )
        self.scoring_model = get_peft_model(scoring_model, self.peft_config)
        
        # load sampling model
        self.reference_tokenizer = load_tokenizer(model_name, dataset, cache_dir)
        reference_model = load_model(model_name, self.device, cache_dir)
        self.reference_model = reference_model
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False

        total = sum(p.numel() for p in self.scoring_model.parameters())
        trainable = sum(p.numel() for p in self.scoring_model.parameters() if p.requires_grad)
        print(f"Trainable / total (parameters): {trainable}/{total}={trainable/total}")
    
    def set_criterion_fn(self, criterion_fn):
        if criterion_fn == "mean":
            self.criterion = 'mean'
            self.criterion_fn = get_sampling_discrepancy_analytic
        else:
            raise ValueError(f"Unknown criterion function: {criterion_fn}")
        
    def print_gradient_requirement(self):
        for name, param in self.named_parameters():
            gradient_requirement = 'Requires Grad' if param.requires_grad else 'Does not require grad'
            color_code = '\033[92m' if param.requires_grad else '\033[91m'  # Green for requires grad, red for does not require grad
            reset_color = '\033[0m'  # Reset color after printing
            print(f"{name}: {color_code}{gradient_requirement}{reset_color}")

    def register_no_grad(self, module_names):
        for name, param in self.named_parameters():
            for selected_module in module_names:
                # print(selected_module, name)
                if selected_module in name:
                    param.requires_grad = False

    def save_pretrained(self, save_directory: str, save_null_distr_only=False):
        """
        Save the scoring model (with LoRA adapter) and all null_distr buffers in Hugging Face format.
        """
        os.makedirs(save_directory, exist_ok=True)

        # 1. 保存 scoring_model (LoRA adapter + 基础模型)
        if not save_null_distr_only:
            scoring_dir = os.path.join(save_directory, "scoring_model")
            self.scoring_model.save_pretrained(scoring_dir, safe_serialization=True)

        # 2. 保存所有 null_distr_* buffers
        null_distrs = {}
        for buffer_name, buffer_value in self.named_buffers():
            if buffer_name.startswith("null_distr_"):
                domain = buffer_name.replace("null_distr_", "")
                null_distrs[domain] = buffer_value.detach().cpu()
        
        if null_distrs:
            torch.save(null_distrs, os.path.join(save_directory, "null_distrs.pt"))
            print(f"✅ Saved {len(null_distrs)} null distributions: {list(null_distrs.keys())}")
        
        # 3. 保存配置信息（包括domain列表）
        config = {
            "domains": list(null_distrs.keys()),
            "criterion": getattr(self, "criterion", None),
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)

        print(f"✅ Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str, *args, **kwargs):
        """
        Load the scoring model, reference model, and all null_distr buffers.
        """
        # 1. 初始化类
        model = cls(*args, **kwargs)

        # 2. 加载 scoring_model
        scoring_dir = os.path.join(load_directory, "scoring_model")
        device = resolve_device(kwargs.get("device", "cuda"))
        model.scoring_model = AutoPeftModelForCausalLM.from_pretrained(
            scoring_dir,
            device_map="auto" if device.startswith("cuda") else None,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        if not device.startswith("cuda"):
            model.scoring_model = model.scoring_model.to(device)

        # 3. 加载所有 null_distr
        null_distrs_path = os.path.join(load_directory, "null_distrs.pt")
        if os.path.exists(null_distrs_path):
            null_distrs = torch.load(null_distrs_path, map_location="cpu")
            for domain, null_distr in null_distrs.items():
                model.set_null_distr(null_distr, domain)
            print(f"✅ Restored {len(null_distrs)} null distributions: {list(null_distrs.keys())}")
        
        # 4. 加载配置信息
        config_path = os.path.join(load_directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            if "criterion" in config and config["criterion"] is not None:
                model.criterion = config["criterion"]
            print(f"✅ Loaded config: {config}")

        print(f"✅ Model loaded from {load_directory}")
        return model
    
    def compute_stats(self, tokenized=None, labels=[""], training_module=False):
        if training_module:
            logits_score = self.scoring_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:]
            logits_ref = self.reference_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:]
            crit, SPO_input  = self.criterion_fn(logits_ref, logits_score, labels)
        else:
            with torch.no_grad(): # get reference
                logits_score = self.scoring_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:] # shape: [bsz, sentence_len, dim]
                logits_ref = self.reference_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:]
                crit, SPO_input = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, SPO_input, logits_score

    def compute_token_level_terms(self, tokenized=None, labels=[""]):
        logits_score = self.scoring_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:]
        logits_ref = self.reference_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:]

        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)
        
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
        return log_likelihood, mean_ref, var_ref

    def forward(self, text, training_module=True):
        original_text = text[0]
        sampled_text = text[1]
        
        tokenized = self.scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:] 
        train_original_crit, _, _ = self.compute_stats(tokenized, labels, training_module=training_module)
        
        tokenized = self.scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:]
        train_sampled_crit, _, _ = self.compute_stats(tokenized, labels, training_module=training_module)
        
        MMDloss = calculate_MMD_loss(train_original_crit, train_sampled_crit)
        output = dict(crit=[train_original_crit.detach(), train_original_crit, train_sampled_crit.detach(), train_sampled_crit], loss=MMDloss)
        return output

    def predict(self, text):
        with torch.inference_mode():
            original_text = text
            
            tokenized = self.scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
            labels = tokenized.input_ids[:, 1:] 
            logits_score = self.scoring_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:] # shape: [bsz, sentence_len, dim]
            logits_ref = self.reference_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:]     

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
            discrepancy = (log_likelihood.mean(dim=-1) - mean_ref.mean(dim=-1))
            var_ref = var_ref.sum(dim=-1).clamp_min(0.0001)
            
        return discrepancy, var_ref

    def set_null_distr(self, null_distr: torch.Tensor, domain: str):
        """
        Set the null distribution tensor safely.
        """
        distr_name = f"null_distr_{domain}"
        self.register_buffer(distr_name, torch.empty(0))

        if not isinstance(null_distr, torch.Tensor):
            null_distr = torch.tensor(null_distr)

        # detach + clone + 移到正确设备
        null_distr = null_distr.detach().clone().to(self.device)

        # 直接覆盖 buffer，避免 delattr 带来的问题
        self._buffers[distr_name] = null_distr
        print(f"✅ Null distribution on {domain} with shape: {self._buffers[distr_name].shape} with mean {self._buffers[distr_name].mean():.4f} and std {self._buffers[distr_name].std():.4f}")

    def compute_p_value(self, text, domain: str):
        """
        Compute p-value for given text using the null distribution of specified domain.
        
        Args:
            text: Input text to compute score for
            domain: Domain name to use for null distribution
        """
        tokenized = self.scoring_tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            return_token_type_ids=False
        ).to(self.device)
        labels = tokenized.input_ids[:, 1:] 
        
        with torch.inference_mode():
            crit, _, _ = self.compute_stats(tokenized, labels, training_module=False)
        
        # 获取对应domain的null distribution
        distr_name = f"null_distr_{domain}"
        if not hasattr(self, distr_name):
            raise ValueError(
                f"No null distribution found for domain '{domain}'. "
                f"Available domains: {self.get_available_domains()}"
            )
        null_distr = getattr(self, distr_name)
        p_value = self.empirical_p_value(crit, null_distr)

        return crit, p_value

    def empirical_p_value(self, crit: torch.Tensor, null_distr: torch.Tensor):
        # Compute p-value: (count + 1) / (total + 1)
        total = null_distr.numel()
        # count = (null_distr >= crit.unsqueeze(-1)).float().sum()   # slow computation
        count = total - torch.searchsorted(null_distr, crit, right=False)[0]
        p_value = (count + 1.0) / (total + 1.0)
        # print(f"p_value (slow): {p_value} & p_value (fast): {(count + 1) / (total + 1)}", )
        return p_value

    def get_available_domains(self):
        """
        Get list of all available domains with null distributions.
        """
        domains = []
        for buffer_name in self._buffers.keys():
            if buffer_name.startswith("null_distr_"):
                domain = buffer_name.replace("null_distr_", "")
                domains.append(domain)
        return domains

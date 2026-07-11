# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch
import numpy as np
import os
from metrics import get_roc_metrics, get_precision_recall_metrics, get_rejection_rate
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from utils import GpuMem
try:
    from transformers import AdamW
except:
    from torch.optim import AdamW

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(model, data, device, verbose=True, track_compute=False):
    model.to(device)
    model.eval()
    eval_loader = DataLoader(data, batch_size=1, shuffle=False)
    epoch_crit_train_original, epoch_crit_train_sampled = [],[]
    time_list = []
    memory_list = []
    tracker = GpuMem()
    with torch.no_grad(), torch.inference_mode():
        for i, batch in enumerate(tqdm.tqdm(eval_loader, desc="Evaluating")):
            text = batch
            if track_compute:
                start = time.perf_counter()
                with tracker:
                    output = model(text, training_module=False)
                time_list.append(time.perf_counter() - start)
                memory_list.append(tracker.memory_usage())
            else:
                output = model(text, training_module=False)
                time_list.append(0.0)
                memory_list.append(0.0)

            epoch_crit_train_original.extend(output['crit'][1].tolist())
            epoch_crit_train_sampled.extend(output['crit'][3].tolist())

            del output
            if i % 20 == 0:  
                torch.cuda.empty_cache()   # 每20步清理一次缓存
        
    fpr, tpr, roc_auc = get_roc_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
    p, r, pr_auc = get_precision_recall_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
    
    if verbose:
        print(f"Total time: {sum(time_list):.4f}s")
        print(f"<Valid> ROC_AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        print(f"<Valid> Real_mean/std: {np.mean(epoch_crit_train_original):.2f}/{np.std(epoch_crit_train_original):.2f}, val_Samples_mean/std: {np.mean(epoch_crit_train_sampled):.2f}/{np.std(epoch_crit_train_sampled):.2f}")
    
    results_dict = {
        "name": "AdaJASAdetectgpt",
        'info': {'n_samples': len(epoch_crit_train_original)},
        'predictions': {'real': epoch_crit_train_original, 
                        'samples': epoch_crit_train_sampled},
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'runtime': time_list, 
        'memory': memory_list,
    }
    return results_dict


def fine_tune(model, data, device, args=None, ckpt_dir='./ckpt',):
    train_loader = DataLoader(data, batch_size=1, shuffle=True)
    epochs = args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs, eta_min=0,
                                  last_epoch=-1)

    scaler = GradScaler()
    model.to(device)

    # Number of iterations for gradient accumulation
    accumulation_steps = args.a
    epoch_losses, i, loss = [], 0, torch.tensor(0.0).to(device)
    epoch_crit_train_original, epoch_crit_train_sampled = [],[]
    print('Fine-tuning model...')
    tracker = GpuMem()
    start = time.perf_counter()
    with tracker:
        for epoch in range(epochs):
            optimizer.zero_grad()
            for batch in tqdm.tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
                text = batch
                scheduler.step()
                with autocast():
                    outputs_1 = model(text)
                    epoch_crit_train_original.extend(outputs_1['crit'][1].tolist())
                    epoch_crit_train_sampled.extend(outputs_1['crit'][3].tolist())
                    loss += (outputs_1['loss'].to(torch.float32)) / accumulation_steps
                
                del outputs_1

                if ((i + 1) % accumulation_steps) == 0:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()

                    if i % 100 == 0:
                        torch.cuda.empty_cache()
                        
                    epoch_losses.append(loss.item())
                    loss = torch.tensor(0.0).to(device)
                epoch_losses.append(loss.item())
                i += 1

            fpr, tpr, roc_auc = get_roc_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
            p, r, pr_auc = get_precision_recall_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
            
            print(f"<Train> ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
            print(f"<Train> Real mean/std: {np.mean(epoch_crit_train_original):.2f}/{np.std(epoch_crit_train_original):.2f}, Samples mean/std: {np.mean(epoch_crit_train_sampled):.2f}/{np.std(epoch_crit_train_sampled):.2f}")
            epoch_avg_loss = np.mean(epoch_losses)
            print(f"<Train> Average Loss for Epoch {epoch}: {epoch_avg_loss}\n")
            epoch_crit_train_original, epoch_crit_train_sampled = [], [] # reset crit
    pre_memory = tracker.memory_usage()
    pre_time = time.perf_counter() - start
    print(f"Total time: {pre_time:.4f}s; Peak memory: {pre_memory:.4f}Gb")

    if args.save_trained:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        model.save_pretrained(ckpt_dir)
        print(f"Saved finetuned model to directory {ckpt_dir}")

    return model


def infer_model(
    model, 
    data, 
    device,
    domain,
):
    test_loader = DataLoader(data, batch_size=1, shuffle=False)
    model.to(device)

    test_original = []
    test_sampled = []
    p_value_original = []
    p_value_sampled = []
    for batch in tqdm.tqdm(test_loader, desc=f"Testing"):
        human_crit, human_p_value = model.compute_p_value(batch[0], domain)
        llm_crit, llm_p_value = model.compute_p_value(batch[1], domain)
        test_original.append(human_crit.item())
        test_sampled.append(llm_crit.item())
        p_value_original.append(human_p_value.item())
        p_value_sampled.append(llm_p_value.item())

    alphas = [0.01, 0.05, 0.1]
    typeI_error = [get_rejection_rate(p_value_original, alpha) for alpha in alphas]
    power = [get_rejection_rate(p_value_sampled, alpha) for alpha in alphas]
    print("alpha      Type-I error      Power")
    for a, t, p in zip(alphas, typeI_error, power):
        print(f"{a:<10.2f}{t:<15.3f}{p:<15.3f}")
    
    results_dict = {
        'info': {'n_samples': len(test_original)},
        'predictions': {'real': test_original, 'samples': test_sampled},
        'inference': {'real': p_value_original, 'samples': p_value_sampled},
        'inference_metrics': {'typeI_error': typeI_error, 'power': power, 'alpha': alphas},
    }

    return results_dict
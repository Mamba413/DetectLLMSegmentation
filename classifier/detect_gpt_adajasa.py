from FineTune.dataset import CustomDataset_rewrite
from FineTune.model import ComputeStat
from FineTune.engine import fine_tune, evaluate_model, set_seed
import torch
from torch.utils.data import Subset
import argparse
import os
import json
import warnings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--a', type=int, default=1, help="accumulation steps")
    parser.add_argument('--epochs', type=int, default=2, help="finetuning epochs")
    parser.add_argument('--ebt', action="store_true", help="Evaluate model before tuning")
    parser.add_argument('--datanum', type=int, default=-1, help="num of training data")
    parser.add_argument('--train_model', action="store_true")
    parser.add_argument('--eval_model', action="store_true")
    parser.add_argument('--save_trained', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--from_pretrained', type=str)
    parser.add_argument('--base_model', type=str, default="gemma-1b")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--train_dataset', type=str, default='./exp_gpt3to4/data/yelp_gemini-2.5-flash&./exp_gpt3to4/data/essay_gemini-2.5-flash&./exp_gpt3to4/data/xsum_gemini-2.5-flash')
    parser.add_argument('--eval_dataset', type=str, default="./exp_gpt3to4/data/writing_gemini-2.5-flash")
    parser.add_argument('--eval_datanum', type=int, default=-1, help="num of valid data")
    parser.add_argument('--output_file', type=str, default="./exp_gpt3to4/results/")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Running with args: {args}")

    if args.from_pretrained:
        print(f"Loading ckpt from {args.from_pretrained}...")
        model = ComputeStat.from_pretrained(args.from_pretrained, args.base_model, args.base_model, cache_dir=args.cache_dir)
    else:
        model = ComputeStat(args.base_model, args.base_model, cache_dir=args.cache_dir)
    model.set_criterion_fn('mean')
    
    if args.train_model:
        train_data = CustomDataset_rewrite(data_json_dir=args.train_dataset) 
        if args.datanum > 0:
            subset_indices = torch.randperm(len(train_data))[:args.datanum]
        else:
            subset_indices = torch.randperm(len(train_data))
            
        train_subset = Subset(train_data, subset_indices)
        print(len(train_subset))

        file_path = os.path.dirname(os.path.abspath(__file__))
        model = fine_tune(
            model, 
            train_subset, 
            device=args.device, 
            args=args,
            ckpt_dir=f"{file_path}/FineTune/ckpt/",
        )
    
    if args.eval_model:
        if args.train_model or args.from_pretrained:
            if '&' in args.eval_dataset:
                eval_file_list = args.eval_dataset.split('&')
            else:
                eval_file_list = [args.eval_dataset]

            for eval_file in eval_file_list:
                val_data = CustomDataset_rewrite(data_json_dir=eval_file)
                if args.eval_datanum > 0:
                    subset_indices = torch.arange(len(val_data))[:args.eval_datanum]
                else:
                    subset_indices = torch.arange(len(val_data))
                val_subset = Subset(val_data, subset_indices)

                results = evaluate_model(model, val_subset, device=args.device)
                output_path = f"{args.output_file}.adajasadetectgpt.json"
                with open(output_path, "w") as f:
                    json.dump(results, f)

                print(f"Evaluation classifier complete.")
                print("=" * 20)
        else:
            warnings.warn("No model trained or loaded. Please set --train_model or --from_pretrained.")
    
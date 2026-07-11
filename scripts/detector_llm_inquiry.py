from pure_llm_detector import llm_inquiry_detector
import json
import argparse
import numpy as np
from utils import load_data
from utils_cp import eval_score
from model import load_model, load_tokenizer
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### parameter for detection model
    parser.add_argument('--base_model', type=str, default="qwen-4b-instruct")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_subsample', type=int, default=-1, help="number of samples for evaluation, -1 means all samples")
    parser.add_argument('--thres_method', type=str, default='cv', choices=['cv', 'clustering'], help="method for binarising scores: cross-fold CV or per-doc k=2 clustering")
    ### parameter for computing details
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--eval_dataset', type=str, default="./exp_location_single_cp/data/squad_mistralai-8b-instruct_rewrite")
    parser.add_argument('--output_file', type=str, default="./exp_location_single_cp/results/squad_mistralai-8b-instruct_rewrite")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args() 
    args.min_sentence = 1
    name = f"llm_inquiry"

    tokenizer = load_tokenizer(args.base_model, cache_dir=args.cache_dir)
    model = load_model(args.base_model, args.device, args.cache_dir)
    model.eval()
    is_gemma = ("gemma-2" in args.base_model.lower())
    is_qwen = ("qwen" in args.base_model.lower())
    if is_gemma:
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    else:
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    ## 1. random selection detected segments and save it into json file to compatible with the existing detection model
    val_data = load_data(args.eval_dataset)
    n_samples = len(val_data['sampled_sentence'])
    if args.num_subsample > 0:
        n_samples = min(n_samples, args.num_subsample)

    results = []
    for i in tqdm(range(n_samples), desc="Processing samples"):
        sens_list = val_data['sampled_sentence'][i]
        sens_label_list = val_data['source_label'][i]
        label_map = {"L": 1, "H": 0}
        sens_label_list = [label_map[x] for x in sens_label_list]
        num_sentence = len(sens_list)
        if num_sentence <= 1:
            continue

        stat_list = llm_inquiry_detector(sens_list, model, tokenizer, args.base_model, args)
        ntokens_list = [len(tokenizer(s, add_special_tokens=False).get("input_ids", [])) for s in sens_list]
        results.append({"labels": sens_label_list, "predictions": stat_list, "ntokens_list": ntokens_list})
    
    ## evaluate detection results
    results = eval_score(results, thres=np.array(0.5))

    # print evaluation results
    class_eval = {'acc': [x["acc"] for x in results], 'rand': [x["rand"] for x in results], 'hausdorff': [x["hausdorff"] for x in results], 'cp_num_diff': [x["cp_num_diff"] for x in results], 'covering': [x["covering"] for x in results], 'tokens_hausdorff': [x["tokens_hausdorff"] for x in results if x["tokens_hausdorff"] is not None], 'wd': [x["wd"] for x in results]}
    print(
        f"Best Accuracy (mean/std): {np.mean(class_eval['acc']):.2f}/{np.std(class_eval['acc']):.2f}",
        f"Covering Metric (mean/std): {np.mean(class_eval['covering']):.2f}/{np.std(class_eval['covering']):.2f}",
        f"CP Number Difference (mean/std): {np.mean(class_eval['cp_num_diff']):.2f}/{np.std(class_eval['cp_num_diff']):.2f}",
    )
    # results
    results_file = f'{args.output_file}.{name}.json'
    results = {
        'name': f'{name}',
        'info': {'n_samples': n_samples},
        'metrics': {
            'acc': np.mean(class_eval['acc']).tolist(), 'acc_std': np.std(class_eval['acc']).tolist(),
            'rand': np.mean(class_eval['rand']).tolist(), 'rand_std': np.std(class_eval['rand']).tolist(),
            'hausdorff': np.mean(class_eval['hausdorff']).tolist(), 'hausdorff_std': np.std(class_eval['hausdorff']).tolist(),
            'cp_num_diff': np.mean(class_eval['cp_num_diff']).tolist(), 'cp_num_diff_std': np.std(class_eval['cp_num_diff']).tolist(),
            'covering': np.mean(class_eval['covering']).tolist(), 'covering_std': np.std(class_eval['covering']).tolist(),
            'tokens_hausdorff': np.mean(class_eval['tokens_hausdorff']).tolist() if class_eval['tokens_hausdorff'] else None,
            'tokens_hausdorff_std': np.std(class_eval['tokens_hausdorff']).tolist() if class_eval['tokens_hausdorff'] else None,
            'wd': np.mean(class_eval['wd']).tolist(), 'wd_std': np.std(class_eval['wd']).tolist(),
        },
        'raw_results': results,
    }
    with open(results_file, 'w') as fout:
        json.dump(results, fout, indent=2)
        print(f'Results written into {results_file}')

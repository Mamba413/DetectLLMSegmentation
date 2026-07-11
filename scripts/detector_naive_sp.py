from pure_llm_detector import LLMDetector
import json
import argparse
import time
import numpy as np
from utils import load_data
from utils_cp import eval_score

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### parameter for detection model
    parser.add_argument('--base_model', type=str, default="gemma-1b")
    parser.add_argument('--aux_model', type=str, default="gemma-1b-instruct")
    parser.add_argument('--from_pretrained', type=str, default="scripts/FineTune/ckpt/")
    parser.add_argument('--phi', type=str, default="FT", help="pure LLM-generated detection method: FT (fine-tuning method) or Bino (binoculars) ", choices=["FT", "Bino", "FDGPT", "GLTR", "Likelihood", "LRR"])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_subsample', type=int, default=-1, help="number of samples for evaluation, -1 means all samples")
    parser.add_argument('--thres_method', type=str, default='clustering', choices=['cv', 'clustering'], help="method for binarising scores: cross-fold CV or per-doc k=2 clustering")
    ### parameter for computing details
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--eval_dataset', type=str, default="./exp_location_single_cp/data/squad_mistralai-8b-instruct_rewrite")
    parser.add_argument('--output_file', type=str, default="./exp_location_single_cp/results/squad_mistralai-8b-instruct_rewrite")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()
    args.min_sentence = 1
    name = f"naive_sp.{args.phi.lower()}"

    # load detection model
    llm_detector = LLMDetector(args)

    ## 1. random selection detected segments and save it into json file to compatible with the existing detection model
    val_data = load_data(args.eval_dataset)
    n_samples = len(val_data['sampled_sentence'])
    if args.num_subsample > 0:
        n_samples = min(n_samples, args.num_subsample)

    results = []
    time_list = []
    for i in tqdm(range(n_samples), desc="Processing samples"):
        sens_list = val_data['sampled_sentence'][i]
        sens_label_list = val_data['source_label'][i]
        label_map = {"L": 1, "H": 0}
        sens_label_list = [label_map[x] for x in sens_label_list]
        num_sentence = len(sens_list)
        if num_sentence <= 1:
            continue

        t0 = time.perf_counter()
        ## sentence-level statistics computation
        stat_list = []
        for j in range(num_sentence):
            start, end = j, j+args.min_sentence
            stat = llm_detector.score(sens_list[start:end][0])
            stat_list.append(stat)
        time_list.append(time.perf_counter() - t0)

        ntokens_list = [len(llm_detector.model.scoring_tokenizer(s, add_special_tokens=False).get("input_ids", [])) for s in sens_list]
        results.append({"labels": sens_label_list, "predictions": stat_list, "ntokens_list": ntokens_list})
    
    ## evaluate detection results
    results = eval_score(results, thres_method=args.thres_method)

    # print evaluation results
    class_eval = {'acc': [x["acc"] for x in results], 'rand': [x["rand"] for x in results], 'hausdorff': [x["hausdorff"] for x in results], 'cp_num_diff': [x["cp_num_diff"] for x in results], 'covering': [x["covering"] for x in results], 'tokens_hausdorff': [x["tokens_hausdorff"] for x in results if x["tokens_hausdorff"] is not None], 'wd': [x["wd"] for x in results]}
    print(
        f"Best Accuracy (mean/std): {np.mean(class_eval['acc']):.2f}/{np.std(class_eval['acc']):.2f}",
        f"Window Difference (mean/std): {np.mean(class_eval['wd']):.2f}/{np.std(class_eval['wd']):.2f}",
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
            'runtime': np.mean(time_list),
            'runtime_std': np.std(time_list),
        },
        'raw_results': results,
    }
    with open(results_file, 'w') as fout:
        json.dump(results, fout, indent=2)
        print(f'Results written into {results_file}')

import json
import argparse
import time
import numpy as np
from utils import load_data
from pure_llm_detector import LLMDetector
from utils_cp import NOT, BinSeg, DPSeg, DPSegSelect, predict_sentence_cp, find_change_points, eval_cp_accuracy, covering_metric
from metrics import get_cp_detection_metrics, window_difference
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### parameter for detection model
    parser.add_argument('--base_model', type=str, default="gemma-1b")
    parser.add_argument('--aux_model', type=str, default="gemma-1b-instruct")
    parser.add_argument('--from_pretrained', type=str, default="scripts/FineTune/ckpt/")
    parser.add_argument('--num_subsample', type=int, default=-1, help="number of samples for evaluation, -1 means all samples")
    ### parameter for change point detection
    parser.add_argument('--phi', type=str, default="FDGPT", help="pure LLM-generated detection method: FT (fine-tuning method) or Bino (binoculars) ", choices=["NFT", "FT", "Bino", "FDGPT", "NFDGPT", "GLTR", "Likelihood", "LRR"])
    parser.add_argument('--cp_method', type=str, default="DP", help="change point detection method", choices=["NOT", "BS", "DP", "Oracle"])
    parser.add_argument('--cp_num', type=int, default=9, help="number of change points")
    parser.add_argument('--cp_selection', type=str, default='fixed', choices=['fixed', 'aicc', 'aic', 'bic'], help="how to choose the number of change points; automatic selection is only supported for --cp_method DP")
    parser.add_argument('--r', type=float, default=1.0, help="Parameters that affacts the number of selected change points. A larger r will lead to fewer change points.")
    parser.add_argument('--max_cp_num', type=int, default=-1, help="maximum number of change points considered when using automatic DP model selection; -1 means use the automatic upper bound")
    parser.add_argument('--M', type=int, default=800, help="number of random intervals per recursion for NOT algorithm")
    parser.add_argument('--weight_type', type=str, default='none', choices=['none', 'ntokens', 'invar'], help="weighting scheme for CP algorithm: ntokens (token-count weights) or invar (inverse-variance weights, requires --phi NFT)")
    parser.add_argument('--power', type=float, default=0.0, help="power parameter for weighted CP detection algorithm. ")
    parser.add_argument('--thres_const', type=float, default=0.5, help="threshold constant for change point detection") # 0.5 for gpt-4o xsum
    parser.add_argument('--thres_method', type=str, default='clustering', choices=['cv', 'clustering'], help="method for binarising segment scores: cross-fold CV or per-doc k=2 clustering")
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--adaptive', action='store_true')
    ### parameter for computing details
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--eval_dataset', type=str, default="./exp_location_multi_cp/data/xsum_gpt-4o_rewrite")
    parser.add_argument('--output_file', type=str, default="./exp_location_multi_cp/results/xsum_gpt-4o_rewrite")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args() 
    print(f"Running with args: {args}")
    args.min_sentence = 1
    args.adaptive = True
    if args.cp_selection != 'fixed' and args.cp_method != 'DP':
        raise ValueError("Automatic change-point selection is only supported when --cp_method DP.")
    if args.weight_type == 'none':
        name = f"naive_cp.{args.phi.lower()}.{args.cp_method.lower()}"
    else:
        name = f"weighted{args.weight_type}{args.power}_cp.{args.phi.lower()}.{args.cp_method.lower()}"
    if args.cp_method == 'DP' and args.cp_selection != 'fixed':
        name = f"{name}.{args.cp_selection.lower()}"

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
        const = args.thres_const
        thres = const * np.sqrt(np.log(num_sentence))
        selected_cp_num = None
        selection_criterion = 'fixed'
        criterion_score = None
        criterion_scores = None

        ## sentence-level statistics computation
        ntokens_list = [len(llm_detector.model.scoring_tokenizer(s, add_special_tokens=False).get("input_ids", [])) for s in sens_list]
        if args.cp_method == "Oracle":
            est_cp = find_change_points(sens_label_list)
            selected_cp_num = len(est_cp)
            selection_criterion = 'oracle'
        else:
            stat_list = []
            var_list = []
            for j in range(num_sentence):
                start, end = j, j + args.min_sentence
                stat = llm_detector.score(sens_list[start:end][0])
                stat_list.append(stat)
                if args.phi in ("NFT", "NFDGPT"):
                    var_list.append(llm_detector.variance)
                else:
                    var_list.append(1.0)  # dummy variance for non-NFT methods

            ## Change points based on the statistics
            if args.weight_type == 'none':
                weights = [1.0 for _ in ntokens_list]
            elif args.weight_type == 'ntokens':
                weights = [n ** args.power for n in ntokens_list]
            elif args.weight_type == 'invar':
                weights = [(1 / v) for v in var_list]
                weights = [x * (y ** args.power) for x, y in zip(weights, ntokens_list)]

            weight_sum = sum(weights)
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                raise ValueError("Change-point weights must have a positive finite sum.")
            weights = [len(weights) * weight / weight_sum for weight in weights]

            # For a 2-sentence document, there is only one valid split point.
            # Avoid calling ruptures on this degenerate case and use the midpoint directly.
            if num_sentence == 2:
                est_cp = [1]
                selected_cp_num = len(est_cp)
                selection_criterion = 'midpoint_short_doc'
            else:
                # cap cp_num to avoid BS/DP requesting more breakpoints than possible
                eff_cp_num = min(args.cp_num, num_sentence - 1)
                if args.cp_method == "BS":
                    est_cp = BinSeg(stat_list, eff_cp_num, w=weights)
                    selected_cp_num = len(est_cp)
                elif args.cp_method == "DP":
                    if args.cp_selection == 'fixed':
                        est_cp = DPSeg(stat_list, eff_cp_num, w=weights)
                        selected_cp_num = len(est_cp)
                    else:
                        # Each segment needs at least ~2 sentences; cap k accordingly.
                        auto_cap = max(1, num_sentence // 2)
                        if args.max_cp_num < 0:
                            max_cp_num = auto_cap
                        else:
                            max_cp_num = min(args.max_cp_num, auto_cap)
                        dp_selection = DPSegSelect(
                            stat_list,
                            w=weights,
                            criterion=args.cp_selection,
                            r=args.r,
                            max_bkps=max_cp_num,
                        )
                        est_cp = dp_selection["est_cp"]
                        selected_cp_num = dp_selection["selected_k"]
                        selection_criterion = dp_selection["criterion"]
                        criterion_score = dp_selection["best_score"]
                        criterion_scores = dp_selection["criterion_scores"]
                else:
                    est_cp = NOT(stat_list, thres, args.M, args.cp_num, w=weights, seed=args.seed, adaptive=args.adaptive)
                    selected_cp_num = len(est_cp)

        ## sentence-level prediction based on the change points
        sentence_preds = predict_sentence_cp(est_cp, sens_list, llm_detector, args.phi, ntokens_list)
        true_cp = find_change_points(sens_label_list)
        eval_true_cp = true_cp + [num_sentence] if len(true_cp) > 0 else [0] + [num_sentence]
        eval_est_cp = est_cp + [num_sentence] if len(est_cp) > 0 else [0] + [num_sentence]
        ri, hau = get_cp_detection_metrics(eval_true_cp, eval_est_cp)
        hau = hau / num_sentence   # normalize to [0, 1]

        time_list.append(time.perf_counter() - t0)
        results.append({"labels": sens_label_list, 'raw_predictions': stat_list, 'weights': weights, "predictions": sentence_preds, 'best_thres': 0.0,
                        "true_cp": true_cp, "est_cp": est_cp,
                        "acc": 0.0, "rand": ri, 
                        "hausdorff": hau,
                        "cp_num_diff": len(true_cp)-len(est_cp), 
                        "selected_cp_num": selected_cp_num if selected_cp_num is not None else len(est_cp),
                        "selection_criterion": selection_criterion,
                        "criterion_score": criterion_score,
                        "criterion_scores": criterion_scores})

    ## Evaluate change points based classification results
    results = eval_cp_accuracy(results, thres_method=args.thres_method)
    for r in results:
        est_label_list = [int(x > r['best_thres']) for x in r['predictions']]
        r['wd'] = window_difference(r['labels'], est_label_list)

    # compute prediction scores for real/sampled passages
    class_eval = {'acc': [x["acc"] for x in results], 
                  'rand': [x["rand"] for x in results], 
                  'hausdorff': [x["hausdorff"] for x in results],
                  'cp_num_diff': [x["cp_num_diff"] for x in results], 
                  'wd': [x["wd"] for x in results],
                  'selected_cp_num': [x["selected_cp_num"] for x in results]}
    print(
        f"Best Accuracy (mean/std): {np.mean(class_eval['acc']):.2f}/{np.std(class_eval['acc']):.2f}",
        f"CP Number Difference (mean/std): {np.mean(class_eval['cp_num_diff']):.2f}/{np.std(class_eval['cp_num_diff']):.2f}",
        f"Window Difference (mean/std): {np.mean(class_eval['wd']):.2f}/{np.std(class_eval['wd']):.2f}",
    )
    # results
    results_file = f'{args.output_file}.{name}.json'
    results = { 
        'name': f'{name}',
        'info': {
            'n_samples': n_samples,
            'cp_selection': args.cp_selection,
            'max_cp_num': args.max_cp_num,
        },
        'metrics': {
            'acc': np.mean(class_eval['acc']).tolist(), 'acc_std': np.std(class_eval['acc']).tolist(),
            'wd': np.mean(class_eval['wd']).tolist(), 'wd_std': np.std(class_eval['wd']).tolist(),
            'selected_cp_num': np.mean(class_eval['selected_cp_num']).tolist(), 'selected_cp_num_std': np.std(class_eval['selected_cp_num']).tolist(),
            'rand': np.mean(class_eval['rand']).tolist(), 'rand_std': np.std(class_eval['rand']).tolist(), 
            'hausdorff': np.mean(class_eval['hausdorff']).tolist(), 'hausdorff_std': np.std(class_eval['hausdorff']).tolist(),
            'cp_num_diff': np.mean(class_eval['cp_num_diff']).tolist(), 'cp_num_diff_std': np.std(class_eval['cp_num_diff']).tolist(),
            'runtime': np.mean(time_list),
            'runtime_std': np.std(time_list),
        },
        'raw_results': results,
    }
    with open(results_file, 'w') as fout:
        json.dump(results, fout, indent=2)
        print(f'Results written into {results_file}')

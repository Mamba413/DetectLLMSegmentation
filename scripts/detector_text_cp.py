from FineTune.engine import set_seed
import json
import argparse
import numpy as np
from utils import load_data
from pure_llm_detector import LLMDetector
from utils_cp import predict_sentence_cp, find_change_points, eval_cp_accuracy
from metrics import get_cp_detection_metrics, covering_metric
from tqdm import tqdm

def NOT_CoAuth(sens_list, detector, c_T, M, n_bkps, seed=None):
    """
    Algorithm 2: Narrowest-Over-Threshold for Co-Authored Text (NOT Co-Auth)

    Inputs
    ------
    sens_list : list[str]
        Sentence list, treated as Y = (Y_1, ..., Y_T).
    c_T : float
        Threshold.
    M : int
        Tuning parameter (# random intervals per recursion).
    model : callable
        Pure LLM-generated text detector φ. Expected usage:
            eval_data = [sens_list[start:end], "placeholder"]
            stat = model(eval_data, training_module=False)['crit'][0]

    Output
    ------
    S : set[int]
        Estimated change-points S ⊂ {1,...,T}, 1-indexed (same convention as NOT()).
    """
    # -------------------- checks --------------------
    if not isinstance(sens_list, (list, tuple)) or not all(isinstance(s, str) for s in sens_list):
        raise TypeError("sens_list must be a list/tuple of strings.")
    T = len(sens_list)
    if T == 0:
        return set()
    if not np.isfinite(c_T):
        raise ValueError("c_T must be a finite number.")
    if not isinstance(M, (int, np.integer)) or M < 0:
        raise ValueError("M must be a nonnegative integer.")
    if M == 0:
        return set()

    if seed is not None:
        np.random.seed(seed)

    # -------------------- precompute per-sentence token lengths --------------------
    # We intentionally ignore cross-sentence boundary tokenization effects:
    # tokens(Y_l..Y_r) ≈ sum_{i=l..r} tokens(Y_i)
    sent_tok_len = np.zeros(T + 1, dtype=np.int64)  # 1-indexed
    for i in range(1, T + 1):
        ids_plain = detector.model.scoring_tokenizer(sens_list[i - 1], add_special_tokens=False).get("input_ids", [])
        sent_tok_len[i] = len(ids_plain)

    prefix_len = np.cumsum(sent_tok_len)  # prefix_len[k] = sum_{i=0..k} sent_tok_len[i], sent_tok_len[0]=0

    def token_len(l, r):
        """Approx #tokens of Y_l..Y_r (1-indexed, inclusive) via sum of per-sentence token counts."""
        if l > r:
            return 0
        return int(prefix_len[r] - prefix_len[l - 1])

    # -------------------- detector φ --------------------
    def phi(l, r):
        crit = detector.score(" ".join(sens_list[(l - 1):r]))
        return float(crit)

    # -------------------- NOT-CoAuth main loop (iterative recursion) --------------------
    rng = np.random.default_rng()
    S = set()
    stack = [(1, T)]  # (s,e), 1-indexed

    while stack and (len(S) < n_bkps):
        s, e = stack.pop()

        # Step 2
        if e - s < 1:
            continue

        # Step 5: sample M intervals (we draw two endpoints in [s,e] and sort;
        # invalid/too-short intervals will be ignored automatically).
        u = rng.integers(s, e + 1, size=M)
        v = rng.integers(s, e + 1, size=M)
        sm = np.minimum(u, v).astype(int)
        em = np.maximum(u, v).astype(int)
        pairs = np.stack([sm, em], axis=1)

        # deduplicate intervals (𝓜 is a set in the algorithm)
        pairs = np.unique(pairs, axis=0)

        sm_arr = pairs[:, 0]
        em_arr = pairs[:, 1]
        M_eff = pairs.shape[0]

        max_stats = np.full(M_eff, -np.inf, dtype=float)
        argmax_b = np.full(M_eff, s, dtype=int)
        interval_tokens = np.full(M_eff, np.iinfo(np.int64).max, dtype=np.int64)

        # Step 9: for each interval m, compute max over b
        for i in range(M_eff):
            sm = int(sm_arr[i])
            em = int(em_arr[i])

            # Need an interior split: sm < b < em => em - sm >= 2
            if em - sm < 2:
                continue

            # token count for (Y_{sm+1},...,Y_{em}) used in Step 13
            interval_tokens[i] = token_len(sm + 1, em)

            best_stat = -np.inf
            best_b = sm + 1

            for b in range(sm + 1, em):  # b in {sm+1,...,em-1}
                # Left: (sm+1..b), Right: (b+1..em)
                T1 = token_len(sm + 1, b)
                T2 = token_len(b + 1, em)
                denom = T1 + T2
                if denom <= 0 or T1 <= 0 or T2 <= 0:
                    stat = 0.0
                else:
                    scale = np.sqrt((T1 * T2) / denom)
                    stat = scale * abs(phi(sm + 1, b) - phi(b + 1, em))

                if stat > best_stat:
                    best_stat = stat
                    best_b = b

            max_stats[i] = best_stat
            argmax_b[i] = best_b

        # Step 9-11: O = {m : max_stats[m] > c_T}
        over = max_stats > c_T
        if not np.any(over):
            continue

        # Step 13: m* = argmin_{m in O} number of tokens of (Y_{sm+1},...,Y_{em})
        masked_tokens = np.where(over, interval_tokens, np.iinfo(np.int64).max)
        m_star = np.where(masked_tokens == np.min(masked_tokens))[0]
        m_star = m_star[np.argmax(max_stats[m_star])]

        # Step 14: b* = argmax ... for interval m*
        b_star = int(argmax_b[m_star])

        # Step 15
        S.add(b_star)

        # Step 16-17
        stack.append((b_star + 1, e))
        stack.append((s, b_star))

    S = sorted(list(S))
    return S


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### parameter for detection model
    parser.add_argument('--base_model', type=str, default="gemma-1b")
    parser.add_argument('--aux_model', type=str, default="gemma-1b-instruct")
    parser.add_argument('--from_pretrained', type=str, default="scripts/FineTune/ckpt/")
    parser.add_argument('--num_subsample', type=int, default=-1, help="number of samples for evaluation, -1 means all samples")
    ### parameter for change point detection
    parser.add_argument('--phi', type=str, default="FT", help="pure LLM-generated detection method: FT (fine-tuning method) or Bino (binoculars) ", choices=["FT", "Bino", "FDGPT"])
    parser.add_argument('--cp_method', type=str, default="NOT", help="change point detection method", choices=["NOT", "BS", "DP"])
    parser.add_argument('--cp_num', type=int, default=3, help="number of change points")
    parser.add_argument('--M', type=int, default=100, help="number of random intervals per recursion for NOT algorithm")
    parser.add_argument('--seed', type=int, default=42)
    ### parameter for computing details
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--eval_dataset', type=str, default="./exp_location/data/squad_mistralai-8b-instruct_rewrite")
    parser.add_argument('--output_file', type=str, default="./exp_location/results/squad_mistralai-8b-instruct_rewrite")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args() 
    args.min_sentence = 1
    print(f"Running with args: {args}")
    name = f"text_cp.{args.phi.lower()}.{args.cp_method.lower()}"
    set_seed(args.seed)

    # load detection model
    llm_detector = LLMDetector(args)

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

        const = 1.0
        thres = const * np.log(num_sentence)

        ##  direct compute segment-level change point
        if args.cp_method == "BS":
            pass
        else:
            est_cp = NOT_CoAuth(sens_list, detector=llm_detector, c_T=thres, M=args.M, n_bkps=args.cp_num, seed=args.seed)

        ## 3. sentence-level prediction based on the change points
        sentence_preds = predict_sentence_cp(est_cp, sens_list, llm_detector)
        true_cp = find_change_points(sens_label_list)
        eval_true_cp = true_cp + [num_sentence] if len(true_cp) > 0 else [0] + [num_sentence]
        eval_est_cp = est_cp + [num_sentence] if len(est_cp) > 0 else [0] + [num_sentence]
        ri, hau = get_cp_detection_metrics(eval_true_cp, eval_est_cp)

        results.append({"labels": sens_label_list, "predictions": sentence_preds, 'best_thres': 0.0,
                        "true_cp": true_cp, "est_cp": est_cp, 
                        "acc": 0.0, "rand": ri, "hausdorff": hau, "cp_num_diff": len(true_cp)-len(est_cp), "covering": covering_metric(sens_label_list, sentence_preds)}) 

    ## Evaluate change points based classification results
    results = eval_cp_accuracy(results)
    
    # compute prediction scores for real/sampled passages
    class_eval = {'acc': [x["acc"] for x in results], 'rand': [x["rand"] for x in results], 'hausdorff': [x["hausdorff"] for x in results], 'cp_num_diff': [x["cp_num_diff"] for x in results], 'covering': [x["covering"] for x in results]}
    print(
        f"Best Accuracy (mean/std): {np.mean(class_eval['acc']):.2f}/{np.std(class_eval['acc']):.2f}", 
        f"Rand Index (mean/std): {np.mean(class_eval['rand']):.2f}/{np.std(class_eval['rand']):.2f}", 
        f"Hausdorff Distance (mean/std): {np.mean(class_eval['hausdorff']):.2f}/{np.std(class_eval['hausdorff']):.2f}", 
        f"CP Number Difference (mean/std): {np.mean(class_eval['cp_num_diff']):.2f}/{np.std(class_eval['cp_num_diff']):.2f}",
        f"Covering Metric (mean/std): {np.mean(class_eval['covering']):.2f}/{np.std(class_eval['covering']):.2f}", 
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
        },    
        'raw_results': results,
    }
    with open(results_file, 'w') as fout:
        json.dump(results, fout, indent=2)
        print(f'Results written into {results_file}')
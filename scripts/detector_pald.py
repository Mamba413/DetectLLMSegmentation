"""
PaLD-TI (Partial-LLM Detector - Text Identification) baseline.
Lei, Hsu, Chen. "PaLD: Detection of Text Partially Written by Large Language Models."
ICLR 2025. https://github.com/jpmorganchase/pald

Algorithm:
  Given sentences x_1, ..., x_n and a T-score function T():
    f_x(S) = T(x[S]) - T(x[S^c])   where x[S] = concatenation of sentences in S
    Ŝ = argmax_{S ⊆ {1,...,n}} f_x(S)
  - Exact for n <= max_exact_n (enumerate all 2^n - 2 subsets)
  - Greedy (Algorithm 1) for n > max_exact_n
"""
import json
import argparse
import itertools
import time
import numpy as np
from utils import load_data
from pure_llm_detector import LLMDetector
from utils_cp import eval_score
from tqdm import tqdm


def _stitch(sentences: list, indices: set) -> str:
    """Concatenate sentences at given indices in order."""
    return " ".join(sentences[i] for i in sorted(indices))


def _f(sentences: list, S: set, detector) -> float:
    """f_x(S) = T(x[S]) - T(x[S^c])."""
    n = len(sentences)
    all_indices = set(range(n))
    S_c = all_indices - S
    t_s = detector.score(_stitch(sentences, S)) if S else 0.0
    t_sc = detector.score(_stitch(sentences, S_c)) if S_c else 0.0
    return t_s - t_sc


def pald_exact(sentences: list, detector) -> set:
    """Exact PaLD-TI: enumerate all 2^n - 2 non-trivial subsets."""
    n = len(sentences)
    best_S = frozenset({0})
    best_score = float('-inf')
    for r in range(1, n):
        for combo in itertools.combinations(range(n), r):
            S = set(combo)
            score = _f(sentences, S, detector)
            if score > best_score:
                best_score = score
                best_S = frozenset(S)
    return best_S


def pald_greedy(sentences: list, detector) -> set:
    """Greedy PaLD-TI (Algorithm 1 in the paper)."""
    n = len(sentences)
    # Initialize S = {e*} where e* = argmax_e f_x({e})
    best_init = max(range(n), key=lambda e: _f(sentences, {e}, detector))
    S = {best_init}
    A = set(range(n)) - S

    current_score = _f(sentences, S, detector)
    while A:
        # Find e' that maximises marginal gain
        gains = {e: _f(sentences, S | {e}, detector) for e in A}
        best_e = max(gains, key=gains.__getitem__)
        new_score = gains[best_e]
        if new_score <= current_score:
            break
        S.add(best_e)
        A.remove(best_e)
        current_score = new_score

    return S


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default="gemma-1b")
    parser.add_argument('--aux_model', type=str, default="gemma-1b-instruct")
    parser.add_argument('--from_pretrained', type=str, default="scripts/FineTune/ckpt/")
    parser.add_argument('--phi', type=str, default="FDGPT",
                        choices=["FT", "Bino", "FDGPT", "GLTR", "Likelihood", "LRR"],
                        help="T-score function to use inside PaLD")
    parser.add_argument('--max_exact_n', type=int, default=10,
                        help="Use exact search for n <= max_exact_n; greedy otherwise")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_subsample', type=int, default=-1)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--eval_dataset', type=str,
                        default="./exp_multi_cp/data/xsum_gpt-4o_rewrite")
    parser.add_argument('--output_file', type=str,
                        default="./exp_multi_cp/results/xsum_gpt-4o_rewrite")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()
    print(f"Running with args: {args}")

    name = f"pald.{args.phi.lower()}"

    llm_detector = LLMDetector(args)

    val_data = load_data(args.eval_dataset)
    n_samples = len(val_data['sampled_sentence'])
    if args.num_subsample > 0:
        n_samples = min(n_samples, args.num_subsample)

    results = []
    time_list = []
    for i in tqdm(range(n_samples), desc="PaLD-TI"):
        sens_list = val_data['sampled_sentence'][i]
        sens_label_list = val_data['source_label'][i]
        label_map = {"L": 1, "H": 0}
        sens_label_list = [label_map[x] for x in sens_label_list]
        num_sentence = len(sens_list)
        if num_sentence <= 1:
            continue

        t0 = time.perf_counter()
        if num_sentence <= args.max_exact_n:
            llm_S = pald_exact(sens_list, llm_detector)
        else:
            llm_S = pald_greedy(sens_list, llm_detector)
        time_list.append(time.perf_counter() - t0)

        # Binary predictions: 1.0 if sentence index in Ŝ, else 0.0
        stat_list = [1.0 if j in llm_S else 0.0 for j in range(num_sentence)]
        results.append({"labels": sens_label_list, "predictions": stat_list})

    # Fixed threshold 0.5 since predictions are already binary
    results = eval_score(results, thres=np.array(0.5))

    class_eval = {
        'acc': [x["acc"] for x in results],
        'rand': [x["rand"] for x in results],
        'hausdorff': [x["hausdorff"] for x in results],
        'cp_num_diff': [x["cp_num_diff"] for x in results],
        'covering': [x["covering"] for x in results],
        'wd': [x["wd"] for x in results],
    }
    print(
        f"Accuracy (mean/std): {np.mean(class_eval['acc']):.3f}/{np.std(class_eval['acc']):.3f}",
        f"WD (mean/std): {np.mean(class_eval['wd']):.3f}/{np.std(class_eval['wd']):.3f}",
    )

    results_file = f'{args.output_file}.{name}.json'
    results = {
        'name': name,
        'info': {'n_samples': n_samples},
        'metrics': {
            'acc': np.mean(class_eval['acc']).tolist(),
            'acc_std': np.std(class_eval['acc']).tolist(),
            'rand': np.mean(class_eval['rand']).tolist(),
            'rand_std': np.std(class_eval['rand']).tolist(),
            'hausdorff': np.mean(class_eval['hausdorff']).tolist(),
            'hausdorff_std': np.std(class_eval['hausdorff']).tolist(),
            'cp_num_diff': np.mean(class_eval['cp_num_diff']).tolist(),
            'cp_num_diff_std': np.std(class_eval['cp_num_diff']).tolist(),
            'covering': np.mean(class_eval['covering']).tolist(),
            'covering_std': np.std(class_eval['covering']).tolist(),
            'wd': np.mean(class_eval['wd']).tolist(),
            'wd_std': np.std(class_eval['wd']).tolist(),
            'runtime': np.mean(time_list),
            'runtime_std': np.std(time_list),
        },
        'raw_results': results,
    }
    with open(results_file, 'w') as fout:
        json.dump(results, fout, indent=2)
        print(f'Results written into {results_file}')

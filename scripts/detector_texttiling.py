"""
TextTiling baseline for mixed-text change-point detection.
Uses NLTK's TextTilingTokenizer (Hearst 1997) to identify topical segment
boundaries, then evaluates those boundaries against true H/L change points.
"""
import json
import argparse
import time
import numpy as np
from utils import load_data
from utils_cp import eval_score, find_change_points
from metrics import get_cp_detection_metrics, window_difference
from tqdm import tqdm

try:
    from nltk.tokenize import TextTilingTokenizer
    import nltk
    # nltk.download('stopwords', quiet=True)
    # nltk.download('punkt', quiet=True)
except ImportError:
    raise ImportError("nltk is required for TextTiling: pip install nltk")


def run_texttiling(sens_list: list, w: int = 20, k: int = 10) -> list:
    """
    Run TextTiling on a list of sentences. Returns estimated change points
    (0-based sentence indices where a new segment begins).

    Parameters
    ----------
    sens_list : list[str]  List of sentences.
    w : int  Pseudosentence size (tokens per pseudosentence).
    k : int  Block comparison size.

    Returns
    -------
    list[int]  0-based change-point indices (start of new segment).
    """
    if len(sens_list) < 2:
        return []

    sep = "\n\n"
    # Record start position of each sentence in the joined text
    sent_starts = []
    pos = 0
    for sent in sens_list:
        sent_starts.append(pos)
        pos += len(sent) + len(sep)
    text = sep.join(sens_list)

    try:
        ttt = TextTilingTokenizer(w=w, k=k)
        tiles = ttt.tokenize(text)
    except Exception:
        # TextTiling may fail on very short texts
        return []

    if len(tiles) <= 1:
        return []

    # Find character positions where tiles end (i.e. tile boundaries)
    tile_end_positions = []
    char_pos = 0
    for tile in tiles[:-1]:
        char_pos += len(tile)
        tile_end_positions.append(char_pos)

    # Map each tile boundary (char pos) to the first sentence that starts at or after it
    change_points = []
    for tb in tile_end_positions:
        for i, sp in enumerate(sent_starts):
            if sp >= tb and i > 0:
                if i not in change_points:
                    change_points.append(i)
                break

    return sorted(set(change_points))


def cp_to_predictions(est_cp: list, num_sentence: int) -> list:
    """
    Convert change points to alternating binary predictions (0/1 per sentence).
    Segment 0 = 0, segment 1 = 1, segment 2 = 0, ...
    The actual orientation (which label wins) is learned by eval_score().
    """
    boundaries = [0] + sorted(est_cp) + [num_sentence]
    preds = [0.0] * num_sentence
    for seg_idx in range(len(boundaries) - 1):
        start, end = boundaries[seg_idx], boundaries[seg_idx + 1]
        label = float(seg_idx % 2)  # alternating 0.0, 1.0, 0.0, ...
        for j in range(start, end):
            preds[j] = label
    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w', type=int, default=20, help="TextTiling pseudosentence size")
    parser.add_argument('--k', type=int, default=10, help="TextTiling block comparison size")
    parser.add_argument('--num_subsample', type=int, default=-1)
    parser.add_argument('--eval_dataset', type=str,
                        default="./exp_multi_cp/data/xsum_gpt-4o_rewrite")
    parser.add_argument('--output_file', type=str,
                        default="./exp_multi_cp/results/xsum_gpt-4o_rewrite")
    args = parser.parse_args()
    print(f"Running with args: {args}")

    name = "texttiling"

    val_data = load_data(args.eval_dataset)
    n_samples = len(val_data['sampled_sentence'])
    if args.num_subsample > 0:
        n_samples = min(n_samples, args.num_subsample)

    results = []
    time_list = []
    for i in tqdm(range(n_samples), desc="TextTiling"):
        sens_list = val_data['sampled_sentence'][i]
        sens_label_list = val_data['source_label'][i]
        label_map = {"L": 1, "H": 0}
        sens_label_list = [label_map[x] for x in sens_label_list]
        num_sentence = len(sens_list)
        if num_sentence <= 1:
            continue

        t0 = time.perf_counter()
        est_cp = run_texttiling(sens_list, w=args.w, k=args.k)
        stat_list = cp_to_predictions(est_cp, num_sentence)
        time_list.append(time.perf_counter() - t0)
        results.append({"labels": sens_label_list, "predictions": stat_list})

    # Use fixed threshold 0.5 since predictions are already in {0, 1}
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

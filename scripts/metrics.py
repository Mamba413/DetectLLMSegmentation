# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.linear_model import LogisticRegression
from ruptures.metrics import randindex, hausdorff
from typing import List, Tuple

# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    if roc_auc < 0.5:
        fpr, tpr, _ = roc_curve([1] * len(real_preds) + [0] * len(sample_preds), real_preds + sample_preds)
        roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_roc_metrics_multi(real_preds, revise_preds, sample_preds):
    label = [0] * len(real_preds) + [1] * len(revise_preds) + [2] * len(sample_preds)
    preds = np.array(real_preds + revise_preds + sample_preds)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    preds = LogisticRegression(random_state=0).fit(preds, label).predict_proba(preds)
    label = label_binarize(label, classes=[0, 1, 2])
    roc_auc = roc_auc_score(label, preds, multi_class='ovo', average='macro')
    return float(roc_auc)

def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds),
                                                  real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    if pr_auc < 0.5:
        precision, recall, _ = precision_recall_curve([1] * len(real_preds) + [0] * len(sample_preds),
                                                      real_preds + sample_preds)
        pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

def get_precision_recall_metrics_multi(real_preds, revise_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(revise_preds) + [2] * len(sample_preds), real_preds + revise_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

def get_rejection_rate(p_values, alpha=0.05):
    return float(np.mean(np.array(p_values) < alpha))

def get_cp_detection_metrics(true_cps, est_cps):
    ri = randindex(true_cps, est_cps)
    hau = hausdorff(true_cps, est_cps)
    return ri, hau

def get_hausdorff_tokenwise(true_cps, est_cps, ntokens_list):
    cumulative_tokens = np.cumsum(ntokens_list)
    token_true_cps = sorted(set(int(cumulative_tokens[cp-1]) for cp in true_cps))
    token_est_cps = sorted(set(int(cumulative_tokens[cp-1]) for cp in est_cps))
    # Need at least one breakpoint each (the final position); return None if degenerate
    if len(token_true_cps) < 1 or len(token_est_cps) < 1:
        return 1  # return max distance to avoid division by zero; will be ignored in analysis
    try:
        hau_tokens = hausdorff(token_true_cps, token_est_cps)
    except Exception:
        return 1
    return hau_tokens / cumulative_tokens[-1]

def get_clustering_threshold(y_score) -> float:
    """Per-document threshold via optimal 1D k=2 clustering.
    Enumerates all split points on sorted unique values, picks the one minimising
    total within-cluster variance (WCSS). Threshold = midpoint of the chosen split.
    With exactly 2 unique values, returns their mean directly.
    """
    unique_vals = np.unique(np.asarray(y_score, dtype=float))
    if len(unique_vals) <= 1:
        return float(unique_vals[0]) if len(unique_vals) == 1 else 0.0
    if len(unique_vals) == 2:
        return float((unique_vals[0] + unique_vals[1]) / 2)
    best_wcss, best_thr = np.inf, unique_vals[0]
    for i in range(1, len(unique_vals)):
        c1 = unique_vals[:i]
        c2 = unique_vals[i:]
        wcss = np.var(c1) * len(c1) + np.var(c2) * len(c2)
        if wcss < best_wcss:
            best_wcss = wcss
            best_thr = float((unique_vals[i - 1] + unique_vals[i]) / 2)
    return best_thr

def get_best_threshold_accuracy(y_true, y_score):
    """
    Compute best accuracy under thresholding, correctly handling ties.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary labels (0/1).
    y_score : array-like of shape (n,)
        Prediction scores (may contain ties).

    Returns
    -------
    best_acc : float
        Maximum achievable accuracy.
    best_thr : float
        Threshold achieving the maximum accuracy.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # sort by score (ascending)
    order = np.argsort(y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    # unique score blocks
    unique_scores, indices = np.unique(y_score, return_index=True)

    n = len(y_true)
    total_pos = y_true.sum()
    total_neg = n - total_pos

    # start with threshold < min(score): predict all 1
    tp = total_pos
    fp = total_neg
    tn = 0
    fn = 0

    best_acc = (tp + tn) / n
    best_thr = unique_scores[0] - 1e-12

    # sweep block by block
    for i in range(len(unique_scores)):
        start = indices[i]
        end = indices[i + 1] if i + 1 < len(indices) else n

        # move this entire block from positive to negative
        block_labels = y_true[start:end]
        tp -= block_labels.sum()
        fn += block_labels.sum()
        fp -= (end - start - block_labels.sum())
        tn += (end - start - block_labels.sum())

        acc = (tp + tn) / n

        # threshold between blocks
        if i + 1 < len(unique_scores):
            thr = 0.5 * (unique_scores[i] + unique_scores[i + 1])
        else:
            thr = unique_scores[i] + 1e-12

        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return best_acc, best_thr

def covering_metric(labels: List[int], est_cp: List[int]) -> float:
    """
    Compute covering metric C(G, G') following Arbeláez et al. (2010).

    Parameters
    ----------
    labels : list[int]
        Ground-truth labels (length T).
    est_cp : list[int]
        Estimated change points.

    Returns
    -------
    float
        Covering score.
    """
    T = len(labels)

    if len(est_cp) == 0:
        return 0.0

    G = segments_from_labels(labels)
    Gp = segments_from_cp(est_cp, T)

    score = []
    for A in G:
        len_A = A[1] - A[0]
        best_jaccard = max(jaccard(A, Ap) for Ap in Gp)
        score.append(len_A * best_jaccard)
    
    score = sum(score) / len(score)
    return score

def jaccard(seg1: Tuple[int, int], seg2: Tuple[int, int]) -> float:
    """
    Jaccard index between two intervals [s1,e1), [s2,e2)
    """
    s1, e1 = seg1
    s2, e2 = seg2

    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter

    return inter / union if union > 0 else 0.0

def segments_from_labels(labels: List[int]) -> List[Tuple[int, int]]:
    """
    Convert label sequence into contiguous segments.
    Returns half-open intervals [start, end).
    """
    segments = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segments.append((start, i))
            start = i
    segments.append((start, len(labels)))
    return segments

def segments_from_cp(est_cp: List[int], T: int) -> List[Tuple[int, int]]:
    """
    Convert change points into segments.
    est_cp are assumed to be 0-based, segment boundaries.
    """
    cp = sorted([c for c in est_cp if 0 < c < T])
    boundaries = [0] + cp + [T]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

def window_difference(reference: List[int], hypothesis: List[int], k: int = None) -> float:
    """
    WindowDifference metric (Pevzner & Hearst, 2002).

    Measures the proportion of sliding windows whose boundary counts differ
    between the reference and hypothesis segmentations. Lower is better and the
    value is bounded in [0, 1].

    Parameters
    ----------
    reference : list[int]
        Ground-truth binary labels (0 or 1), length T.
    hypothesis : list[int]
        Estimated binary labels (0 or 1), length T.
    k : int, optional
        Window size. Default: T // (2 * max(1, n_reference_segments)).

    Returns
    -------
    float
        Fraction of windows where the boundary counts disagree.
    """
    T = len(reference)
    if T < 2:
        return 0.0

    ref = np.asarray(reference, dtype=int)
    hyp = np.asarray(hypothesis, dtype=int)

    # Boundary indicators: 1 where adjacent labels differ (length T-1)
    ref_bounds = (ref[1:] != ref[:-1]).astype(int)
    hyp_bounds = (hyp[1:] != hyp[:-1]).astype(int)

    if k is None:
        n_segs = max(1, int(ref_bounds.sum()) + 1)
        k = max(1, T // (2 * n_segs))

    n_windows = T - k  # number of windows of size k in the (T-1)-length boundary array
    if n_windows <= 0:
        return 0.0

    # Prefix sums: ref_prefix[i] = sum(ref_bounds[0:i])
    ref_prefix = np.concatenate([[0], np.cumsum(ref_bounds)])
    hyp_prefix = np.concatenate([[0], np.cumsum(hyp_bounds)])

    # ref_counts[i] = number of boundaries in window [i, i+k)
    ref_counts = ref_prefix[k:] - ref_prefix[:n_windows]
    hyp_counts = hyp_prefix[k:] - hyp_prefix[:n_windows]

    return float(np.mean(ref_counts != hyp_counts))

if __name__ == "__main__":
    # synthetic example
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_score = np.array([2.4115355014801025, 2.4115355014801025, 2.4115355014801025, 2.4115355014801025, -2.2309446334838867, -2.2309446334838867, -2.2309446334838867, -2.2309446334838867, -2.2309446334838867, -2.2309446334838867])

    acc, thr = get_best_threshold_accuracy(y_true, y_score)

    print("Best accuracy:", acc)
    print("Best threshold:", thr)

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    est_cp = [5]
    cover = covering_metric(y_true, est_cp)
    print("Covering metric:", cover)

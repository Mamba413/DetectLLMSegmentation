import numpy as np
import ruptures as rpt
from metrics import get_best_threshold_accuracy, get_clustering_threshold, get_cp_detection_metrics, covering_metric, get_hausdorff_tokenwise, window_difference
from sklearn.metrics import accuracy_score
from ruptures.utils import sanity_check

def eval_score(results, thres=None, thres_method='cv'):
    n_samples = len(results)
    half_samples = n_samples >> 1
    if thres is None and thres_method == 'cv':
        half_results_1 = results[:half_samples]
        half_results_2 = results[half_samples:len(results)]
        _, best_thres1 = get_best_threshold_accuracy([y for x in half_results_1 for y in x['labels']], [y for x in half_results_1 for y in x['predictions']])
        _, best_thres2 = get_best_threshold_accuracy([y for x in half_results_2 for y in x['labels']], [y for x in half_results_2 for y in x['predictions']])
    elif thres is not None:
        best_thres1 = thres
        best_thres2 = thres
    new_results = []
    for i in range(n_samples):
        if thres_method == 'clustering' and thres is None:
            best_thres = get_clustering_threshold(results[i]['predictions'])
        elif i < half_samples:
            best_thres = best_thres2
        else:
            best_thres = best_thres1
        num_sentence = len(results[i]['labels'])
        est_label_list = [int(x > best_thres) for x in results[i]['predictions']]
        acc = accuracy_score(results[i]['labels'], est_label_list)
        true_cp, est_cp = find_change_points(results[i]['labels']), find_change_points(est_label_list)
        eval_true_cp = true_cp + [num_sentence] if len(true_cp) > 0 else [0] + [num_sentence]
        eval_est_cp = est_cp + [num_sentence] if len(est_cp) > 0 else [0] + [num_sentence]
        ri, hau = get_cp_detection_metrics(eval_true_cp, eval_est_cp)
        hau = hau / num_sentence   # normalize to [0, 1]
        cover = covering_metric(results[i]['labels'], est_label_list)
        if not isinstance(hau, float):
            hau = hau.item()
        if not isinstance(cover, float):
            cover = cover.item()
        if not isinstance(best_thres, float):
            best_thres = best_thres.item()
        if 'ntokens_list' in results[i]:
            tokens_hau = get_hausdorff_tokenwise(eval_true_cp, eval_est_cp, results[i]['ntokens_list'])
        else:
            tokens_hau = None
        wd = window_difference(results[i]['labels'], [int(x) for x in est_label_list])
        new_results.append({
            "labels": results[i]['labels'], "predictions": results[i]['predictions'], "best_thres": best_thres, "true_cp": true_cp, "est_cp": est_cp,
            "acc": acc, "rand": ri, "hausdorff": hau, 'cp_num_diff': len(true_cp) - len(est_cp), "covering": cover, "tokens_hausdorff": tokens_hau, "wd": wd})
    return new_results

def eval_cp_accuracy(results, thres_method='cv'):
    n_samples = len(results)
    half_samples = n_samples >> 1
    if thres_method == 'cv':
        half_results_1 = results[:half_samples]
        half_results_2 = results[half_samples:len(results)]
        _, best_thres1 = get_best_threshold_accuracy([y for x in half_results_1 for y in x['labels']], [y for x in half_results_1 for y in x['predictions']])
        _, best_thres2 = get_best_threshold_accuracy([y for x in half_results_2 for y in x['labels']], [y for x in half_results_2 for y in x['predictions']])
    for i in range(n_samples):
        if thres_method == 'clustering':
            best_thres = get_clustering_threshold(results[i]['predictions'])
        elif i < half_samples:
            best_thres = best_thres2
        else:
            best_thres = best_thres1
        est_label_list = [int(x > best_thres) for x in results[i]['predictions']]
        acc = accuracy_score(results[i]['labels'], est_label_list)
        results[i]['best_thres'] = float(best_thres) if not isinstance(best_thres, float) else best_thres
        results[i]['acc'] = acc
    return results

def find_change_points(labels):
    """
    Find change points in a deterministic label sequence.

    Parameters
    ----------
    labels : list[str]
        Sequence of labels, e.g., ["L", "L", "H", "H", "L"]

    Returns
    -------
    cps : list[int]
        Change point indices (0-based), where a new segment starts.
    """
    cps = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            cps.append(i)
    return cps

def predict_sentence_cp(cp, sentence_list, llm_detector, detector_name, ntokens_list):
    '''
    predict whether a sentence is a change point based on the cp indices
    1. convert sentence_list into segments based on cp
    2. use model to predict each segment
    3. put the prediction results back to sentence level
    '''
    n = len(sentence_list)
    sentence_preds = [0] * n

    boundaries = [0] + cp + [n]

    # ------------------------------------------------
    # 2. segment-wise prediction
    # ------------------------------------------------
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]

        segment_sents = sentence_list[start:end]
        if len(segment_sents) == 0:
            continue
        score = llm_detector.score(" ".join(segment_sents))

        # ------------------------------------------------
        # 3. put results back to sentence level
        # ------------------------------------------------
        for j in range(start, end):
            if detector_name in ("NFT", "NFDGPT"):
                sentence_preds[j] = score * sum(ntokens_list[start:end]) / np.sqrt(llm_detector.variance)
            else:
                sentence_preds[j] = score

    return sentence_preds

def max_cusum_statistic(prefix_w, prefix_wy, s_m, e_m, M, min_size=2, min_cusum_segment=0):
    max_stats = np.empty(M, dtype=float)
    argmax_b = np.empty(M, dtype=int)
    for i in range(M):
        sm = int(s_m[i])
        em = int(e_m[i])

        L = em - sm
        if L < min_size:
            max_stats[i] = -np.inf
            argmax_b[i] = sm
            continue

        b = np.arange(sm + min_cusum_segment, em - min_cusum_segment + 1, dtype=int)
        if len(b) == 0:
            max_stats[i] = -np.inf
            argmax_b[i] = sm
            continue

        W1 = prefix_w[b] - prefix_w[sm]      # sum_{t=sm+1}^b w_t
        W2 = prefix_w[em] - prefix_w[b]      # sum_{t=b+1}^{em} w_t
        WY1 = prefix_wy[b] - prefix_wy[sm]   # sum_{t=sm+1}^b w_t Y_t
        WY2 = prefix_wy[em] - prefix_wy[b]   # sum_{t=b+1}^{em} w_t Y_t

        valid = (W1 > 0) & (W2 > 0)
        stat = np.full(b.shape[0], -np.inf, dtype=float)
        if np.any(valid):
            mu1 = WY1[valid] / W1[valid]
            mu2 = WY2[valid] / W2[valid]
            scale = np.sqrt((W1[valid] * W2[valid]) / (W1[valid] + W2[valid]))
            stat[valid] = scale * np.abs(mu1 - mu2)

        j = int(np.argmax(stat))  # first maximizer if ties
        max_stats[i] = float(stat[j])
        argmax_b[i] = int(b[j])
    return max_stats, argmax_b

def NOT(Y, c_T, M, n_bkps, w=None, seed=None, min_size=4, adaptive=False, min_cusum_segment=0):
    """
    Python implementation of Algorithm 1 (NOT).

    Input
    -----
    Y   : array-like, shape (T,)
          data vector (Y_1, ..., Y_T)'
    c_T : float
          threshold
    M   : int
          tuning parameter (# random intervals per recursion)
    adaptive : bool
          if True, adaptively update c_T using observed CUSUM peaks

    Output
    ------
    S : set of int
        estimated change-points S ⊂ {1, ..., T} (1-indexed)
    """
    if seed is not None:
        np.random.seed(seed)

    y0 = np.asarray(Y, dtype=float).reshape(-1)
    T = y0.size
    if T == 0:
        return set()
    if not isinstance(M, (int, np.integer)) or M < 0:
        raise ValueError("M must be a nonnegative integer.")
    if not np.isfinite(c_T):
        raise ValueError("c_T must be a finite number.")

    if w is None:
        w0 = np.ones(T, dtype=float)
    else:
        w0 = np.asarray(w, dtype=float).reshape(-1)
        if w0.size != T:
            raise ValueError("w must have the same length as Y.")
        if not np.all(np.isfinite(w0)):
            raise ValueError("w must be finite.")
        if np.any(w0 < 0):
            raise ValueError("w must be nonnegative (negative weights are not supported).")

    # Use 1-indexed convention to match the algorithm exactly.
    y = np.zeros(T + 1, dtype=float)
    y[1:] = y0
    ww = np.zeros(T + 1, dtype=float)
    ww[1:] = w0

    prefix_w = np.cumsum(ww)
    prefix_wy = np.cumsum(ww * y)  # prefix[t] = sum_{i=1}^t Y_i

    cusum_peaks = []
    c_T_min = c_T
    c_T_current = c_T

    rng = np.random.default_rng()
    S = set()
    stack = [(1, T)]  # (s, e)

    while stack and (len(S) < n_bkps):
        s, e = stack.pop()

        # Step 2: if e - s <= 1 then STOP
        if e - s <= 1:
            continue

        # Step 5-7: draw M intervals; if M == 0 then STOP
        if M == 0:
            continue

        # Draw M pairs (u,v) uniformly from {s,...,e} with u != v, then sort -> (s_m, e_m)
        # and remove deduplicate intervals (𝓜 is a set in the algorithm)
        u = rng.integers(s, e + 1, size=M)
        v = rng.integers(s, e + 1, size=M)
        s_m = np.minimum(u, v).astype(int)
        e_m = np.maximum(u, v).astype(int)
        pairs = np.unique(np.stack([s_m, e_m], axis=1), axis=0)
        s_m = pairs[:, 0]
        e_m = pairs[:, 1]
        M_eff = pairs.shape[0]

        # Step 9 (i): compute max CUSUM statistic on each interval
        max_stats, argmax_b = max_cusum_statistic(prefix_w, prefix_wy, s_m, e_m, M_eff, min_size, min_cusum_segment)

        # Step 9 (ii): define O as those exceeding c_T_current
        over = max_stats > c_T_current
        # Step 10-11: if O == ∅ then STOP
        if not np.any(over):
            continue

        # Step 13: m* = argmin_{m in O} (sum_{t=1}^{e_m} w_t - sum_{t=1}^{s_m} w_t)
        # i.e., minimal total weight on (s_m, e_m] when weights are one
        interval_w = (prefix_w[e_m] - prefix_w[s_m]).astype(float)
        interval_w_masked = np.where(over, interval_w, np.inf)
        m_star = np.where(interval_w_masked == np.min(interval_w_masked))[0]
        # interval_len = (e_m - s_m).astype(float)
        # interval_len_masked = np.where(over, interval_len, np.inf)
        # m_star = np.where(interval_len_masked == np.min(interval_len_masked))[0]

        m_star = m_star[np.argmax(max_stats[m_star])]

        # Step 14: b* := argmax ... on interval m*
        b_star = int(argmax_b[m_star])

        # Step 15: S := S ∪ {b*}
        if all([abs(s_tmp - b_star) >= min_size for s_tmp in S]):
            S.add(b_star)
            if adaptive:
                peak_val = float(max_stats[m_star])
                cusum_peaks.append(peak_val)
                if len(cusum_peaks) >= 3:
                    c_T_current = max(c_T_min, float(np.mean(cusum_peaks) - 2.0 * np.std(cusum_peaks)))
            # Step 16-17: recurse on [s, b*] and [b*+1, e]
            stack.append((b_star + 1, e))
            stack.append((s, b_star))
        else:
            continue

    S = sorted(list(S))
    return S

def BinSeg(Y, n_bkps, w=None):
    if isinstance(Y, list):
        Y = np.array(Y)
    algo = rpt.Binseg(model="l2", min_size=1, jump=2).fit(Y, weight=w)
    S = algo.predict(n_bkps=n_bkps)
    return S[:-1]  # remove T

def DPSeg(Y, n_bkps, w=None):
    if isinstance(Y, list):
        Y = np.array(Y)
    algo = rpt.Dynp(model="l2", min_size=1, jump=2).fit(Y, weight=w)
    S = algo.predict(n_bkps=n_bkps)
    return S[:-1]  # remove T

def _get_information_criterion(rss, n_samples, n_bkps, criterion, r, eps=1e-12):
    """Compute an information criterion from the weighted residual sum of squares."""
    criterion = criterion.lower()
    n_segments = n_bkps + 1
    p = 2 * n_segments
    rss_safe = max(float(rss) / max(int(n_samples), 1), eps)
    base = n_samples * np.log(rss_safe)
    if criterion == "aic":
        return base + r * 2 * p
    if criterion == "bic":
        return base + r * p * np.log(n_samples)
    if criterion == "aicc":
        aic = base + r * 2 * p
        denom = n_samples - p - 1
        if denom <= 0:
            return np.inf
        return aic + (2 * p * (p + 1)) / denom
    raise ValueError(f"Unsupported information criterion: {criterion}")

def DPSegSelect(Y, w=None, criterion="aicc", r=1.0, max_bkps=None):
    """Select the number of breakpoints for Dynp via an information criterion."""
    if isinstance(Y, list):
        Y = np.array(Y)
    Y = np.asarray(Y)
    n_samples = int(Y.shape[0])
    min_size = 1
    jump = 2
    if n_samples == 0:
        return {
            "est_cp": [],
            "selected_k": 0,
            "best_score": None,
            "criterion": criterion.lower(),
            "criterion_scores": [],
        }

    algo = rpt.Dynp(model="l2", min_size=min_size, jump=jump).fit(Y, weight=w)

    auto_max_bkps = max(0, min(n_samples - 1, n_samples // min_size - 1))
    if max_bkps is None:
        max_bkps = auto_max_bkps
    else:
        max_bkps = max(0, min(int(max_bkps), auto_max_bkps))

    candidates = []
    criterion_scores = []
    for n_bkps in range(max_bkps + 1):
        if not sanity_check(n_samples=n_samples, n_bkps=n_bkps, jump=jump, min_size=min_size):
            criterion_scores.append({
                "cp_num": int(n_bkps),
                "score": None,
                "rss": None,
            })
            continue
        partition = algo.seg(0, n_samples, n_bkps)
        rss = float(sum(partition.values()))
        candidates.append({
            "n_bkps": n_bkps,
            "partition": partition,
            "rss": rss,
            "score_record_idx": len(criterion_scores),
        })
        criterion_scores.append({
            "cp_num": int(n_bkps),
            "score": None,
            "rss": rss,
        })

    if len(candidates) == 0:
        raise ValueError(
            f"No feasible DP segmentation for n_samples={n_samples}, min_size={min_size}, jump={jump}, max_bkps={max_bkps}."
        )

    requested_criterion = criterion.lower()
    used_criterion = requested_criterion
    for candidate in candidates:
        candidate["score"] = _get_information_criterion(
            candidate["rss"], n_samples, candidate["n_bkps"], requested_criterion, r
        )

    finite_requested = [c for c in candidates if np.isfinite(c["score"])]
    if len(finite_requested) == 0 and requested_criterion == "aicc":
        used_criterion = "aic"
        for candidate in candidates:
            candidate["score"] = _get_information_criterion(
                candidate["rss"], n_samples, candidate["n_bkps"], used_criterion, r
            )

    valid_candidates = [c for c in candidates if np.isfinite(c["score"])]
    if len(valid_candidates) == 0:
        raise ValueError(f"No finite {used_criterion} scores were found for DP segmentation.")

    for candidate in candidates:
        criterion_scores[candidate["score_record_idx"]]["score"] = (
            float(candidate["score"]) if np.isfinite(candidate["score"]) else None
        )

    best = min(valid_candidates, key=lambda x: (x["score"], x["n_bkps"]))
    bkps = sorted(end for _, end in best["partition"].keys())
    if len(bkps) > 0 and bkps[-1] == n_samples:
        bkps = bkps[:-1]

    return {
        "est_cp": bkps,
        "selected_k": int(best["n_bkps"]),
        "best_score": float(best["score"]),
        "criterion": used_criterion,
        "criterion_scores": criterion_scores,
    }

if __name__ == "__main__":
    import matplotlib.pylab as plt

    n = 40  # number of samples
    n_bkps, sigma = 3, 1.0  # number of change points, noise standard deviation
    signal, bkps = rpt.pw_constant(n, 1, n_bkps, noise_std=sigma, seed=0)
    bkps = bkps[:-1]  # remove n

    # --- 2) run Binary Segmentation (fixed k=3) ---
    my_bkps = BinSeg(signal, n_bkps=n_bkps)

    # --- 3) run NOT ---
    c_T = 1   # threshold (tune this; smaller => more detections)
    M = 10000      # number of random intervals per recursion
    not_cps = NOT(signal, c_T=c_T, M=M, n_bkps=3)
    not_cps = sorted(not_cps)

    print("True change-points (1-indexed):", bkps)
    print("NOT estimated change-points:", not_cps)
    print(f"Binary segmentation (L2, k={n_bkps}) estimated change-points:", my_bkps)

    # --- 4) plot ---
    t = np.arange(1, len(signal) + 1)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(t, signal, label="Observed Y")

    for cp in bkps:
        ax.axvline(cp, color="black", linestyle="--", label="True change-point" if cp == bkps[0] else None)
    for cp in not_cps:
        ax.axvline(cp, color="red", linestyle=":", label="NOT estimate" if cp == not_cps[0] else None)
    for cp in my_bkps:
        ax.axvline(cp, color="green", linestyle="-.", label="BinSeg (L2) estimate" if cp == my_bkps[0] else None)

    ax.set_title("NOT vs Binary Segmentation on a 3-change-point synthetic signal")
    ax.set_xlabel("t")
    ax.set_ylabel("Y_t")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

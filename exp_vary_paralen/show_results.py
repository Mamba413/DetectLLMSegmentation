"""
exp_vary_paralen/show_results.py

Reads result files from exp_vary_paralen/results/ and prints:
  - A Markdown table (stdout)
  - A LaTeX table (stdout)

The table follows the paragraph-length layout:
  - columns are paragraph lengths k=2,3,4,5,6
  - rows are Voting, SenPred, Textilling, VCP, and WCP
  - cells report WD mean with a 95% CI half-width

Expected filename patterns:
  Voting:
    <dataset>_<model>_<task>_k<K>.voting_sp.<phi>.json
  SenPred:
    <dataset>_<model>_<task>_k<K>.naive_sp.<phi>.json
  Textilling:
    <dataset>_<model>_<task>_k<K>.texttiling.json
  VCP:
    <dataset>_<model>_<task>_k<K>.naive_cp.<phi>.dp.json
  WCP:
    <dataset>_<model>_<task>_k<K>.weightedntokens<P>_cp.<phi>.dp.json

Run from the repo root:
    python exp_vary_paralen/show_results.py
or from inside exp_vary_paralen/:
    python show_results.py
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "results")

METHOD_LABELS = {
    "voting_sp": "Voting",
    "naive_sp": "SenPred",
    "texttiling": "Textilling",
    "naive_cp": "VCP",
    "weightedntokens_cp": "WCP",
}

METHOD_ORDER = ("Voting", "SenPred", "Textilling", "VCP", "WCP")

RESULT_PAT = re.compile(
    r"(?P<dataset>[^_]+)_(?P<model>.+?)_(?P<task>[^_]+)_k(?P<k>\d+)"
    r"\.(?P<method>voting_sp|naive_sp|texttiling|naive_cp|weightedntokens[0-9.]+_cp)"
    r"(?:\.(?P<phi>[^.]+))?(?:\.dp)?\.json$"
)

DATASET_LABELS = {
    "writing": "Writing",
    "govreport": "GovReport",
}


def load_results(result_dir: str) -> dict:
    """
    Returns:
        data[dataset][method_label][k] -> list of per-sample result dicts
    """
    data: dict = {}

    for fpath in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        fname = os.path.basename(fpath)
        match = RESULT_PAT.search(fname)
        if not match:
            continue

        method = match.group("method")
        if method.startswith("weightedntokens"):
            method_key = "weightedntokens_cp"
        else:
            method_key = method

        with open(fpath) as f:
            res = json.load(f)
        raw_results = res.get("raw_results")
        if not raw_results:
            continue

        label = METHOD_LABELS[method_key]
        dataset = match.group("dataset")
        k = int(match.group("k"))
        data.setdefault(dataset, {}).setdefault(label, {})[k] = raw_results

    return data


def metric_values(raw_results: list[dict], key: str) -> list[float]:
    return [float(item[key]) for item in raw_results if key in item]


def standard_error(vals: list[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def confidence_interval(vals: list[float]) -> float:
    return 1.96 * standard_error(vals)


def build_rows(data: dict) -> tuple[list[int], list[dict]]:
    ks = sorted({k for method_data in data.values() for k in method_data})
    rows = []

    for method in METHOD_ORDER:
        row = {"method": method}
        for k in ks:
            vals = metric_values(data.get(method, {}).get(k, []), "wd")
            row[k] = {
                "mean": float(np.mean(vals)) if vals else None,
                "ci": confidence_interval(vals) if vals else 0.0,
            }
        rows.append(row)

    return ks, rows


def best_by_k(rows: list[dict], ks: list[int]) -> dict[int, float | None]:
    best = {}
    for k in ks:
        vals = [
            row[k]["mean"]
            for row in rows
            if row.get(k, {}).get("mean") is not None
        ]
        best[k] = min(vals) if vals else None
    return best


def fmt_plain(mean: float, ci: float) -> str:
    return f"{mean:.2f}±{ci:.2f}"


def fmt_tex(mean: float, ci: float) -> str:
    return f"{mean:.2f}" + r"{\scriptsize$\pm$" + f"{ci:.2f}" + "}"


def print_markdown(title: str, ks: list[int], rows: list[dict]) -> None:
    best = best_by_k(rows, ks)

    print(f"\n### {title}\n")
    print("| Method | " + " | ".join(str(k) for k in ks) + " |")
    print("| --- | " + " | ".join(["---"] * len(ks)) + " |")
    for row in rows:
        cells = [row["method"]]
        for k in ks:
            mean = row[k]["mean"]
            ci = row[k]["ci"]
            if mean is None:
                cells.append("--")
                continue
            cell = fmt_plain(mean, ci)
            if best[k] is not None and abs(mean - best[k]) < 1e-9:
                cell = f"**{cell}**"
            cells.append(cell)
        print("| " + " | ".join(cells) + " |")


def print_latex(title: str, ks: list[int], rows: list[dict], label: str) -> None:
    best = best_by_k(rows, ks)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{title}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "c" * len(ks) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\textbf{{{k}}}" for k in ks) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        cells = []
        for k in ks:
            mean = row[k]["mean"]
            ci = row[k]["ci"]
            if mean is None:
                cells.append("--")
                continue
            cell = fmt_tex(mean, ci)
            if best[k] is not None and abs(mean - best[k]) < 1e-9:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(row["method"] + " & " + " & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    print("\n% --- LaTeX ---")
    print("\n".join(lines))


def main() -> None:
    data = load_results(RESULT_DIR)
    if not data:
        print(f"No result files found in {RESULT_DIR}")
        print("Expected patterns:")
        print("  <dataset>_<model>_<task>_k<K>.voting_sp.<phi>.json")
        print("  <dataset>_<model>_<task>_k<K>.naive_sp.<phi>.json")
        print("  <dataset>_<model>_<task>_k<K>.texttiling.json")
        print("  <dataset>_<model>_<task>_k<K>.naive_cp.<phi>.dp.json")
        print("  <dataset>_<model>_<task>_k<K>.weightedntokens<P>_cp.<phi>.dp.json")
        return

    for dataset, dataset_data in sorted(data.items()):
        ks, rows = build_rows(dataset_data)
        dataset_label = DATASET_LABELS.get(dataset, dataset)
        title = (
            f"{dataset_label}: WD results under different paragraph-length "
            "settings (mean with 95\\% CI)."
        )
        print_markdown(title, ks, rows)
        print_latex(title, ks, rows, label=f"tab:vary_paralen_{dataset}")


if __name__ == "__main__":
    main()

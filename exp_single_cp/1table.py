import json
import math
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict

# -------------------------
# Configuration
# -------------------------
TASKS = ["continuation"]
DATASETS = ["squad", "xsum", "writing"]
DATASET_LABELS = {"squad": "WikiQA", "xsum": "News", "writing": "Story"}
MODELS = ["claude-haiku-4-5"]
PRINT_MODELS = {
    "claude-haiku-4-5": "Claude4.5",
}

# method key → (display name, filename builder)
# filename builder receives (dataset, model, task)
def _fname(suffix):
    return _fname_first(suffix)


def _fname_first(*suffixes):
    def builder(d, m, tasks):
        for task in tasks:
            for suffix in suffixes:
                candidate = f"{d}_{m}_{task}.{suffix}.json"
                if os.path.exists(os.path.join(BASE_DIR, candidate)):
                    return candidate
        return f"{d}_{m}_{tasks[0]}.{suffixes[0]}.json"
    return builder

METHODS = [
    'llm_inquiry',
    'pald',
    "texttiling",
    "naive_sp",
    "voting_sp",
    "naive_cp_aic",
    "weightedinvar2.0_cp_aic",
]
PRINT_METHOD = {
    'llm_inquiry':               "LLMPred",
    'pald':                      "PaLD",
    "texttiling":                "TextTiling",
    "naive_sp":                  "SenPred",
    "voting_sp":                 "Voting",
    "naive_cp_aic":              "VCP",
    "weightedinvar2.0_cp_aic":   "WCP",
}
FNAME_BUILDER = {
    "llm_inquiry":               _fname("llm_inquiry"),
    "pald":                      _fname("pald"),
    "texttiling":                _fname("texttiling"),
    "naive_sp":                  _fname_first("naive_sp.ft", "naive_sp.fdgpt"),
    "voting_sp":                  _fname_first("voting_sp.ft", "voting_sp.fdgpt"),
    "naive_cp_aic":               _fname_first("naive_cp.ft.dp.aic", "naive_cp.nfdgpt.dp.aic"),
    "weightedinvar2.0_cp_aic":    _fname_first("weightedinvar2.0_cp.ft.dp.aic", "weightedinvar2.0_cp.nfdgpt.dp.aic"),
}

METRIC_KEYS = ["wd", "cp_num_diff"]
METRIC_LABELS = {
    "wd":          r"WD $\downarrow$",
    "cp_num_diff": r"CE $\downarrow$",
}

BASE_DIR = "./results"
missing_files = []


def standard_error(vals):
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var) / math.sqrt(len(vals))


def metric_summary(res, metric):
    metrics = res["metrics"]
    raw_results = res.get("raw_results") or []
    vals = [
        float(item[metric])
        for item in raw_results
        if item.get(metric) is not None
    ]
    if vals:
        return sum(vals) / len(vals), standard_error(vals)
    return metrics.get(metric), None


# -------------------------
# Load results
# -------------------------
results = defaultdict(lambda: defaultdict(dict))

for model in MODELS:
    for method in METHODS:
        for dataset in DATASETS:
            fname = FNAME_BUILDER[method](dataset, model, TASKS)
            fpath = os.path.join(BASE_DIR, fname)
            if not os.path.exists(fpath):
                missing_files.append(fpath)
                continue
            with open(fpath) as f:
                res = json.load(f)
            results[model][method][dataset] = {}
            for k in METRIC_KEYS:
                mean, se = metric_summary(res, k)
                results[model][method][dataset][k] = mean
                results[model][method][dataset][f"{k}_se"] = se

ACTIVE_MODELS = [
    model for model in MODELS
    if any(results[model].get(method, {}) for method in METHODS)
]
ACTIVE_METHODS = [
    method for method in METHODS
    if any(results[model].get(method, {}) for model in ACTIVE_MODELS)
]
ACTIVE_DATASETS = DATASETS
if not ACTIVE_MODELS:
    ACTIVE_MODELS = MODELS
if not ACTIVE_METHODS:
    ACTIVE_METHODS = METHODS


# -------------------------
# Find best value per (model, dataset, metric)
# -------------------------
def is_better(metric, a, b):
    """Return True if a is strictly better than b."""
    if metric == "acc":
        return a > b
    elif metric == "cp_num_diff":
        return abs(a) < abs(b)
    else:
        return a < b

best = defaultdict(lambda: defaultdict(dict))  # best[model][dataset][metric]
for model in ACTIVE_MODELS:
    for dataset in ACTIVE_DATASETS:
        for metric in METRIC_KEYS:
            vals = [
                results[model][method][dataset][metric]
                for method in ACTIVE_METHODS
                if dataset in results[model].get(method, {})
                and results[model][method][dataset].get(metric) is not None
            ]
            if not vals:
                best[model][dataset][metric] = None
                continue
            best_val = vals[0]
            for v in vals[1:]:
                if is_better(metric, v, best_val):
                    best_val = v
            best[model][dataset][metric] = best_val


# -------------------------
# Helpers
# -------------------------
def fmt_bold(x, se, metric, model, dataset):
    if x is None:
        return "--"
    s = f"{x:.2f}"
    if se is not None:
        s += r"{\scriptsize$\pm$" + f"{se:.2f}" + "}"
    bv = best[model][dataset].get(metric)
    if bv is not None and abs(x - bv) < 1e-9:
        return rf"\textbf{{{s}}}"
    return s


# -------------------------
# Print LaTeX table
# -------------------------
n_metrics = len(METRIC_KEYS)
n_datasets = len(ACTIVE_DATASETS)

# column spec: ll | cc | cc | cc
col_spec = "ll|" + "|".join(["c" * n_metrics] * n_datasets)

print(r"\begin{table}[H]")
print(r"\centering")
print(
    r"\caption{Detection results across methods, datasets and models. "
    r"Best results per column are in \textbf{bold}.}"
)
print(r"\label{tab:single_cp_results}")
print(r"\small")
print(r"\setlength{\tabcolsep}{4pt}")
print(r"\renewcommand{\arraystretch}{1.15}")
print(rf"\begin{{tabular}}{{{col_spec}}}")
print(r"\toprule")

# Header row 1
ds_headers = []
for i, ds in enumerate(ACTIVE_DATASETS):
    sep = "|" if i < n_datasets - 1 else ""
    ds_headers.append(rf"\multicolumn{{{n_metrics}}}{{c{sep}}}{{{DATASET_LABELS[ds]}}}")
print(
    r"\multirow{2}{*}{Model} & \multirow{2}{*}{Method} & "
    + " & ".join(ds_headers)
    + r" \\"
)

# Header row 2
metric_header = " & ".join(METRIC_LABELS[k] for k in METRIC_KEYS)
print("& & " + " & ".join([metric_header] * n_datasets) + r" \\")
print(r"\midrule")

# Data rows
for model in ACTIVE_MODELS:
    print(rf"\multirow{{{len(ACTIVE_METHODS)}}}{{*}}{{{PRINT_MODELS[model]}}}")
    for method in ACTIVE_METHODS:
        row = ["", PRINT_METHOD[method]]
        for dataset in ACTIVE_DATASETS:
            m = results[model].get(method, {}).get(dataset, {})
            for metric in METRIC_KEYS:
                row.append(fmt_bold(m.get(metric), m.get(f"{metric}_se"), metric, model, dataset))
        print(" & ".join(row) + r" \\")
    if model != ACTIVE_MODELS[-1]:
        print(r"\midrule")
    else:
        print(r"\bottomrule")

print(r"\end{tabular}")
print(r"\end{table}")

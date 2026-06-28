# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.linear_model import LogisticRegression

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


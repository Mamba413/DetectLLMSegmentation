import os
import json
import numpy as np
import matplotlib.pyplot as plt


result_path = "./exp_longdocs/results/"

methods = [
    "binoculars",
    "roberta-large-openai-detector",
    "sampling_discrepancy_analytic",
    "adajasadetectgpt",
]

METHOD_NAME_MAP = {
    "roberta-large-openai-detector": "Roberta (Gehrmann et al. 2019)",
    "binoculars": "Binoculars (Hans et al. 2024)",
    "sampling_discrepancy_analytic": "FastDetectGPT (Bao et al. 2024)",
    "adajasadetectgpt": "AdaDetectGPT (Zhou et al. 2025)",
}

color_palette = [
    "#E15759",  # red
    "#F28E2B",  # orange
    "#59A14F",  # green
    "#4E79A7",  # blue
]
METHOD_COLORS = {key: color_palette[i] for i, key in enumerate(methods)}

datasets_map = {
    "squad_gemini-2.5-flash_polish": "WikiQA",
    "writing_gemini-2.5-flash_expand": "Story",
}

yy_values = [5, 10, 20, 40, 80, 160, 320]
shown_yy_values = yy_values


def load_results():
    """
    results[method][dataset] = list of AUC over yy_values
    """
    results = {m: {} for m in methods}

    for method in methods:
        for ds in datasets_map.keys():
            auc_list = []

            for yy in yy_values:
                filename = f"{result_path}/{ds}_{yy}.{method}.json"
                if not os.path.exists(filename):
                    print(f"[WARN] File not found: {filename}")
                    auc_list.append(np.nan)
                    continue

                with open(filename, "r") as f:
                    data = json.load(f)

                auc = data["metrics"]["roc_auc"]
                auc_list.append(auc)

            results[method][ds] = auc_list

    return results


# -------------------------------
# 绘图
# -------------------------------
def plot_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)

    x_log = np.log2(yy_values)

    for ax, (ds_key, ds_name) in zip(axes, datasets_map.items()):
        for method in methods:
            auc_list = results[method][ds_key]

            ax.plot(
                x_log,
                auc_list,
                marker="o",
                linewidth=3,
                markersize=8.8,
                color=METHOD_COLORS[method],
                label=METHOD_NAME_MAP[method],
            )

        ax.set_title(ds_name, fontsize=18, weight="bold")
        ax.set_xlabel("Length of texts", fontsize=15, weight="bold")

        ax.set_xticks(x_log)
        ax.set_xticklabels(shown_yy_values, fontsize=13)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("AUC", fontsize=15, weight="bold")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    handles, labels = axes[0].get_legend_handles_labels()

    legend_prop = {
        'weight': 'bold',
        'size': 16
    }
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        prop=legend_prop,
        frameon=False,
    )
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(
        "exp_longdocs/length_vs_auc_methods.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    results = load_results()
    plot_results(results)

"""
All graphs for the app (FR-09): class distribution,
model comparison and confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BG = "#b7a07f"        # same as the page background
INK = "#3D3A4B"
FAKE = "#C47A7A"
GENUINE = "#4F9C86"


# common figure setup
def _base(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)
    return fig, ax


# fr-09: fake vs genuine bar chart
def class_distribution_chart(dist, title):
    fig, ax = _base((4.5, 3.2))
    bars = ax.bar(list(dist.keys()), list(dist.values()), color=[FAKE, GENUINE])
    ax.bar_label(bars, color=INK)
    ax.set_title(title, color=INK)
    return fig


# fr-09: accuracy, precision, recall and F1 for all models
def model_comparison_chart(metrics):
    names = list(metrics.keys())
    items = [("accuracy", "Accuracy", "#4F9C86"), ("precision", "Precision", "#8E7CA8"),
             ("recall", "Recall", "#C47A7A"), ("f1", "F1-score", "#D9A441")]
    fig, ax = _base((8, 3.8))
    x = np.arange(len(names))
    width = 0.2
    for i, (key, label, color) in enumerate(items):
        ax.bar(x + i * width, [metrics[n][key] * 100 for n in names],
               width, label=label, color=color)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names])
    # y-axis starts at 80 so small differences are visible
    ax.set_ylim(80, 100)
    ax.set_ylabel("Score (%)", color=INK)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    return fig


# fr-09: confusion matrix heatmap
def confusion_matrix_chart(cm, title):
    cmap = LinearSegmentedColormap.from_list("soft", ["#D9CBB5", "#4F9C86"])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    fig.patch.set_facecolor(BG)
    ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fake", "Genuine"], color=INK)
    ax.set_yticklabels(["Fake", "Genuine"], color=INK)
    ax.set_xlabel("Predicted", color=INK)
    ax.set_ylabel("Actual", color=INK)
    ax.set_title(title, color=INK, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", color=INK, fontsize=12)
    return fig
# The static PNG figures used in the research report

from pathlib import Path

import matplotlib

# The Agg backend draws straight to files, so no display is needed. It has to be chosen before
# pyplot is imported, which is why the imports below are out of the usual order.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

# Colours used for the highlighted lines, in order: blue, orange, aqua, yellow, magenta, green,
# violet, red. Everything that is not highlighted is drawn in the muted grey.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED = "#9aa0a6"
INK = "#1f2328"
INK_2 = "#59636e"


# Applies the same plot styling to every figure
def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c9ced4",
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e8ebef",
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "font.size": 9,
        }
    )


_style()


# Gives each highlighted series its own colour and a thicker line, and every other series a thin
# muted grey line. Returns name -> (colour, line width, alpha).
def _series_colors(names, highlight):
    line_styles = {}
    for name in names:
        if name in highlight:
            line_styles[name] = (PALETTE[highlight.index(name) % len(PALETTE)], 2.0, 1.0)
        else:
            line_styles[name] = (MUTED, 1.0, 0.7)
    return line_styles


# The majority baseline is drawn dashed, because it often sits exactly on top of buy and hold
def _dash(name):
    return (0, (4, 2)) if name == "majority" else "-"


# Saves a figure to disk and closes it to free the memory
def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# Cumulative return, drawdown and rolling volatility figures for several strategies at once
def plot_equity_panels(
    returns, highlight, path_prefix, title_suffix, rolling_window=63, periods=252
):
    names = list(returns)

    # Highlighted lines are drawn last so they sit on top of the grey ones
    draw_order = [name for name in names if name not in highlight] + [
        name for name in highlight if name in names
    ]
    line_styles = _series_colors(names, highlight)
    paths = []

    # Cumulative return
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for name in draw_order:
        colour, line_width, alpha = line_styles[name]
        cumulative = (1 + returns[name].fillna(0)).cumprod() - 1
        ax.plot(
            cumulative.index,
            cumulative.values * 100,
            color=colour,
            lw=line_width,
            alpha=alpha,
            ls=_dash(name),
            label=name if name in highlight else None,
        )
    ax.plot([], [], color=MUTED, lw=1, label="other models")
    ax.axhline(0, color="#c9ced4", lw=0.8)
    ax.set_ylabel("cumulative net return (%)")
    ax.set_title(f"Cumulative net return after costs — {title_suffix}")
    ax.legend(loc="upper left", ncol=2)
    paths.append(_save(fig, Path(f"{path_prefix}_cumulative.png")))

    # Drawdown
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for name in draw_order:
        colour, line_width, alpha = line_styles[name]
        equity = (1 + returns[name].fillna(0)).cumprod()
        drawdown = equity / equity.cummax() - 1
        ax.plot(
            drawdown.index,
            drawdown.values * 100,
            color=colour,
            lw=line_width,
            alpha=alpha,
            label=name if name in highlight else None,
        )
    ax.set_ylabel("drawdown (%)")
    ax.set_title(f"Drawdown — {title_suffix}")
    ax.legend(loc="lower left", ncol=2)
    paths.append(_save(fig, Path(f"{path_prefix}_drawdown.png")))

    # Rolling volatility
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for name in draw_order:
        colour, line_width, alpha = line_styles[name]
        rolling_vol = returns[name].rolling(rolling_window).std() * np.sqrt(periods)
        ax.plot(
            rolling_vol.index,
            rolling_vol.values * 100,
            color=colour,
            lw=line_width,
            alpha=alpha,
            label=name if name in highlight else None,
        )
    ax.set_ylabel("annualised volatility (%)")
    ax.set_title(f"Rolling {rolling_window}-day volatility — {title_suffix}")
    ax.legend(loc="upper left", ncol=2)
    paths.append(_save(fig, Path(f"{path_prefix}_rolling_vol.png")))
    return paths


# A grid of confusion matrices, one per model. The colours are normalised by row, and each cell
# is labelled with its count and its row percentage.
def plot_confusion_matrices(cms, path, title):
    num_models = len(cms)
    num_cols = min(3, num_models)
    num_rows = int(np.ceil(num_models / num_cols))
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(3.3 * num_cols, 3.0 * num_rows), squeeze=False
    )

    for ax, (name, matrix) in zip(axes.ravel(), cms.items(), strict=False):
        row_share = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        labels = np.array(
            [[f"{matrix[i, j]}\n{row_share[i, j]:.0%}" for j in range(2)] for i in range(2)]
        )
        sns.heatmap(
            row_share,
            annot=labels,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            ax=ax,
            xticklabels=["pred down", "pred up"],
            yticklabels=["actual down", "actual up"],
            linewidths=2,
            linecolor="white",
        )
        ax.set_title(name, fontsize=9)
        ax.grid(False)

    # Hiding any empty panels left over in the grid
    for ax in axes.ravel()[num_models:]:
        ax.axis("off")
    fig.suptitle(title, fontweight="bold", color=INK)
    return _save(fig, path)


# ROC curve for each model, along with the diagonal line a random guess would follow
def plot_roc_curves(curves, highlight, path, title):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color="#c9ced4", lw=1, ls="--", label="chance")
    for i, (name, (outcomes, probabilities)) in enumerate(curves.items()):

        # A ROC curve needs both classes to be present
        if len(np.unique(outcomes)) < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(outcomes, probabilities)
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(outcomes, probabilities)
        is_selected = name == highlight
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            color=PALETTE[i % len(PALETTE)],
            lw=2.2 if is_selected else 1.2,
            label=f"{name} (AUC {auc:.3f}){' ← selected' if is_selected else ''}",
        )
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_aspect("equal")
    return _save(fig, path)


# Calibration plot (using quantile bins) with a histogram of the predicted probabilities underneath
def plot_calibration(curves, path, title, n_bins=10):
    fig, (ax, hist_ax) = plt.subplots(
        2, 1, figsize=(5.6, 6.4), gridspec_kw={"height_ratios": [3, 1.2]}, sharex=False
    )

    # lowest and highest track the range covered so the diagonal line can be sized to fit
    lowest, highest = 1.0, 0.0
    for i, (name, (outcomes, probabilities)) in enumerate(curves.items()):

        # Models that always predict the same probability have nothing to plot
        if np.unique(probabilities).size < 2:
            continue
        observed_rate, mean_predicted = calibration_curve(
            outcomes, probabilities, n_bins=n_bins, strategy="quantile"
        )
        lowest = min(lowest, mean_predicted.min(), observed_rate.min())
        highest = max(highest, mean_predicted.max(), observed_rate.max())
        ax.plot(
            mean_predicted,
            observed_rate,
            marker="o",
            ms=4,
            color=PALETTE[i % len(PALETTE)],
            lw=1.4,
            label=name,
        )
        hist_ax.hist(
            probabilities, bins=40, histtype="step", color=PALETTE[i % len(PALETTE)], lw=1.2
        )

    padding = 0.02
    ax.plot(
        [lowest - padding, highest + padding],
        [lowest - padding, highest + padding],
        color="#c9ced4",
        ls="--",
        lw=1,
        label="perfect calibration",
    )
    ax.set_xlabel("mean predicted P(up)")
    ax.set_ylabel("observed up-day frequency")
    ax.set_title(title)
    ax.legend(fontsize=7)
    hist_ax.set_xlabel("predicted P(up)")
    hist_ax.set_ylabel("count")
    return _save(fig, path)


# Horizontal bar charts with one panel per importance table.
# tables maps each panel title to (DataFrame, name of the column to plot).
def plot_importance(tables, path, title, top=15):
    num_panels = len(tables)
    fig, axes = plt.subplots(
        1, num_panels, figsize=(4.4 * num_panels, 0.32 * top + 1.4), squeeze=False
    )
    for ax, (panel_title, (df, value_column)) in zip(axes.ravel(), tables.items(), strict=False):

        # The top features by absolute size, reversed so the biggest ends up at the top
        largest_first = df[value_column].abs().sort_values(ascending=False).index
        top_features = df.reindex(largest_first).head(top).iloc[::-1]

        # Positive values are blue and negative values are orange
        bar_colours = [
            PALETTE[0] if value >= 0 else PALETTE[1] for value in top_features[value_column]
        ]
        ax.barh(top_features["feature"], top_features[value_column], color=bar_colours, height=0.6)
        if "importance_std" in top_features and value_column == "importance_mean":
            ax.errorbar(
                top_features[value_column],
                top_features["feature"],
                xerr=top_features["importance_std"],
                fmt="none",
                ecolor=INK_2,
                lw=0.8,
            )
        ax.axvline(0, color="#c9ced4", lw=0.8)
        ax.set_title(panel_title, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(title, fontweight="bold", color=INK)
    return _save(fig, path)

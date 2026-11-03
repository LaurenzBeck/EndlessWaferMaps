"""📊 Visualization utilities for training metrics and paper figures."""

import matplotlib.pyplot as plt
import plotille
import seaborn as sns
import torch
from matplotlib.figure import Figure


def plot_epoch_losses(history: list[dict[str, float]], width: int = 114) -> None:
    if not history:
        return

    epochs = [int(row["epoch"]) for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]
    epoch_span = max(epochs) - min(epochs)
    plot_width = max(20, width - 36)
    figure = plotille.Figure()
    figure.width = plot_width
    figure.height = 14
    figure.x_label = "epoch"
    figure.y_label = "loss"
    figure.with_colors = False
    figure.origin = False
    if epoch_span > 0:
        figure.set_x_limits(min(epochs), max(epochs))
    figure.x_ticks_fkt = lambda value, _: round(value)
    figure.plot(epochs, train_losses, label="train", marker="T")
    figure.plot(epochs, val_losses, label="eval", marker="E")
    rendered_plot = figure.show(legend=False)
    print("loss: T=train, E=eval")
    print(rendered_plot)


def plot_epoch_learning_rate(history: list[dict[str, float]], width: int = 114) -> None:
    if not history:
        return

    epochs = [int(row["epoch"]) for row in history]
    learning_rates = [row["lr"] for row in history]
    epoch_span = max(epochs) - min(epochs)
    plot_width = max(20, width - 36)
    figure = plotille.Figure()
    figure.width = plot_width
    figure.height = 14
    figure.x_label = "epoch"
    figure.y_label = "learning rate"
    figure.with_colors = False
    figure.origin = False
    if epoch_span > 0:
        figure.set_x_limits(min(epochs), max(epochs))
    figure.x_ticks_fkt = lambda value, _: round(value)
    figure.plot(epochs, learning_rates, label="lr")
    rendered_plot = figure.show(legend=False)
    print(rendered_plot)


def plot_epoch_metrics(history: list[dict[str, float]], width: int = 114) -> None:
    if not history:
        return

    metric_names = sorted({name for row in history for name in row if name not in {"epoch", "train_loss", "val_loss", "lr"}})
    if not metric_names:
        return

    epochs = [int(row["epoch"]) for row in history]
    markers = ["P", "R", "M", "A", "B", "C", "D", "F"]
    epoch_span = max(epochs) - min(epochs)
    plot_width = max(20, width - 36)
    figure = plotille.Figure()
    figure.width = plot_width
    figure.height = 14
    figure.x_label = "epoch"
    figure.y_label = "metric"
    figure.with_colors = False
    figure.origin = False
    if epoch_span > 0:
        figure.set_x_limits(min(epochs), max(epochs))
    figure.x_ticks_fkt = lambda value, _: round(value)
    legend_parts = []
    for index, metric_name in enumerate(metric_names):
        marker = markers[index % len(markers)]
        figure.plot(epochs, [row[metric_name] for row in history], label=metric_name, marker=marker)
        legend_parts.append(f"{marker}={metric_name}")
    rendered_plot = figure.show(legend=False)
    print(f"evaluation metrics: {', '.join(legend_parts)}")
    print(rendered_plot)


def set_paper_style(*, use_latex: bool = True, double_column: bool = False) -> None:
    """Apply the paper-quality publication style.

    Call once at the top of every notebook. The default figure size is a
    single-column figure; use ``double_column=True`` for WACV double-column
    figures. ``use_latex=True`` renders text via TeX with Computer Modern Roman.
    """
    figure_size = (6.875, 2.25) if double_column else (3.5, 2.25)

    sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update(
        {
            "figure.figsize": figure_size,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )
    if use_latex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            },
        )


def plot_cardinality_confusion_matrix(
    confusion_matrix: torch.Tensor,
    **kwargs: dict,
) -> Figure:
    """Plots a cardinality confusion matrix using matplotlib.

    Args:
        confusion_matrix (torch.Tensor): confusion matrix (e.g. produced by torchmetrics).
        kwargs (dict): additional keyword arguments are passed to `ax.imshow`.

    Returns:
        Figure: matplotlib figure instance.
    """
    cmat = confusion_matrix.clone()
    cmat = cmat / cmat.sum(dim=1, keepdims=True)

    f = plt.figure()
    ax = f.add_subplot(111)

    ax.imshow(cmat.cpu().numpy(), interpolation="nearest", cmap="Blues", **kwargs)
    plt.grid(False)

    plt.ylabel("true cardinality")
    plt.xlabel("predicted cardinality")

    return f


def plot_multilabel_confusion_matrix(
    confusion_matrix: torch.Tensor,
    **kwargs,
) -> Figure:
    """Plots a multilabel confusion matrix using matplotlib.

    The rows are internally normalized to one to better compare classes in the case of a class imbalance.

    Args:
        confusion_matrix (torch.Tensor): confusion matrix (e.g. produced by torchmetrics).
        kwargs (dict): additional keyword arguments are passed to `ax.imshow`.

    Returns:
        Figure: matplotlib figure instance.
    """
    cmat = confusion_matrix.clone()
    cmat = cmat / cmat.sum(dim=1, keepdims=True)

    f = plt.figure()
    ax = f.add_subplot(111)

    ax.imshow(cmat.cpu().numpy(), interpolation="nearest", cmap="Blues", **kwargs)
    plt.grid(False)

    plt.ylabel("true label")
    plt.xlabel("predicted label")

    return f

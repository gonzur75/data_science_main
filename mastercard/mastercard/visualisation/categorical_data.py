import math

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def dist_categorical_visualisation(
        df: pd.DataFrame,
        categorical_cols: list[str],
        n_cols: int = 2,
        figsize: tuple | None = None,
        title: str | None = "Categorical Feature Distributions",
        save_path: str | None = None
) -> tuple[plt.Figure, dict[str, plt.Axes]] | None:
    """
    Plots count distributions for multiple categorical columns using subplots, in a clean and professional style.

    Args:
        df: DataFrame containing the data.
        categorical_cols: List of column names to visualize.
        n_cols: Number of columns in the subplot grid.
        figsize: Size of the entire figure. If None, inferred automatically.
        title: Title of the whole figure.
        save_path: If provided, saves the figure to this path instead of displaying it.
    """

    n = len(categorical_cols)
    n_rows = math.ceil(n / n_cols)
    figsize = figsize or (6 * n_cols, 4.5 * n_rows)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(f"Distribution of {col}", fontsize=13, weight="bold")
        ax.set_xlabel("Count", fontsize=11)
        ax.set_ylabel(col, fontsize=11)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)
        max_val = df[col].value_counts().max()
        ax.set_xlim(0, max_val * 1.1)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if title:
        fig.suptitle(title, fontsize=16, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig, axes
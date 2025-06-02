import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def dist_visualisation(
        data: np.ndarray | pd.Series | list,
        title: str = "",
        show: bool = True
) -> tuple[plt.Figure, dict[str, plt.Axes]] | None:
    """
    Visualize the distribution of data using boxplot, violinplot, and histogram.

    Args:
        data: Input numerical data to visualize.
        title: Title of the entire figure.
        show: If True, displays the plot. If False, returns the figure and axes.

    Returns:
        Tuple of matplotlib figure and axes dictionary if show=False, otherwise None.
    """
    fig = plt.figure(constrained_layout=True)
    plt.style.use('seaborn-v0_8')

    ax_dict = fig.subplot_mosaic([
        ['boxplot', 'violin'],
        ['histogram', 'violin'],
        ['histogram', 'violin'],
    ])

    mean_props = dict(color="darkorange", linestyle="dashed")

    ax_dict['boxplot'].boxplot(
        data,
        showmeans=True,
        meanline=True,
        meanprops=mean_props,
        vert=False
    )

    violin = ax_dict['violin'].violinplot(
        data,
        showmeans=True,
        showmedians=True
    )
    violin["cbars"].set_linestyle('dotted')

    ax_dict['histogram'].hist(data, bins='sqrt', color='orange')

    ax_dict['violin'].set_xlabel('KDE')
    ax_dict['histogram'].set_ylabel('Count Frequency')

    fig.suptitle(title)

    if not show:
        return fig, ax_dict

    plt.show()
    return None
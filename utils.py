import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_DIR = "data"
PLOTS_DIR = "results"


def plot_hist(
    data: pd.Series,
    title: str,
    ylabel: str,
    fig_title: str,
    show: bool = True,
) -> None:
    plt.hist(data, bins=np.arange(len(data.unique()) + 1) - 0.5)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fig_title))

    if show:
        plt.show()


def plot_hist_stacked(
    data: pd.Series,
    outcome: pd.Series,
    title: str,
    fig_title: str,
    figsize: tuple = (),
    species: list = [],
    show: bool = True,
) -> None:
    if not species:
        species = sorted(data.unique(), key=lambda el: el is np.nan)
    else:
        assert set(species) == set(data.unique())
    weight_counts = {}

    for category in outcome.unique():
        weight_counts[category] = np.array(
            [
                (
                    data[outcome == category] == value
                    if value is not np.nan
                    else data[outcome == category].isna()
                ).sum()
                for value in species
            ]
        )

    if not figsize:
        _, ax = plt.subplots()
    else:
        _, ax = plt.subplots(figsize=figsize)

    bottom = np.zeros(len(species))
    width = 0.5

    species = ["NA" if el is np.nan else el for el in species]
    colors = np.linspace((0.45, 0.65, 1.0), (0.0, 0.0, 0.24), len(weight_counts))

    for (key, weight_count), color in zip(weight_counts.items(), colors):
        ax.bar(species, weight_count, width, label=key, bottom=bottom, color=color)
        bottom += weight_count

    ax.set_title(title)
    ax.legend()

    if len(species) >= 6:
        plt.xticks(rotation=45)
    if len(species) >= 9:
        plt.xticks(rotation="vertical")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fig_title))

    if show:
        plt.show()

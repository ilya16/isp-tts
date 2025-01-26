from enum import Enum
from typing import Union, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import torch


def check_array_type(array):
    if isinstance(array, torch.Tensor):
        array = array.data.cpu().numpy()
    return array


class FigureOrders(Enum):
    row = "row"
    col = "col"


def plot_figure(
        data: Union[torch.Tensor, dict[str, torch.Tensor]],
        figsize=(12, 3),
        xlabel=None,
        ylabel=None,
        title=None,
        order=FigureOrders.row
):
    if not isinstance(data, dict):
        data = {None: data}

    n_rows = len(data)
    width, height = figsize

    if order == FigureOrders.row:
        fig, axes = plt.subplots(nrows=n_rows, figsize=(width, height * n_rows))
    else:
        fig, axes = plt.subplots(ncols=n_rows, figsize=(width * n_rows, height))

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    if title is not None:
        fig.suptitle(title)

    for ax, (name, value) in zip(axes, data.items()):
        value = check_array_type(value)

        im = ax.imshow(value, aspect="auto", origin="lower", interpolation='none')
        plt.colorbar(im, ax=ax)

        if name is not None:
            ax.set_title(name)

    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    plt.tight_layout()

    fig.canvas.draw()
    plt.close()

    return fig


def plot_attention(alignment: Union[torch.Tensor, dict[str, torch.Tensor]], title=None):
    return plot_figure(
        alignment,
        figsize=(6, 4),
        xlabel="Decoder time step",
        ylabel="Encoder time step",
        title=title
    )


def plot_spectrogram(spectrogram: Union[torch.Tensor, dict[str, torch.Tensor]], title=None):
    return plot_figure(
        spectrogram,
        figsize=(12, 3),
        xlabel="Frames",
        ylabel="Channels",
        title=title
    )

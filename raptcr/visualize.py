from abc import ABC, abstractmethod
import joblib
from pathlib import Path
from typing import Union

import colorcet as cc
import numpy as np
import pandas as pd
from umap import ParametricUMAP
from umap.parametric_umap import load_ParametricUMAP
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.stats import gaussian_kde
from .hashing import Cdr3Hasher
from .analysis import TcrCollection


class BaseVisualizer(ABC):
    """
    Abstract class for visualization.
    """

    def __repr__(self) -> str:
        return "RapTCR Visualization"

    @abstractmethod
    def save(self, filepath: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def from_file(self, filepath: str):
        ...

    @abstractmethod
    def plot(self, data: TcrCollection):
        ...


class ParametricUmapTransformer(BaseVisualizer):
    def __init__(self, hasher: Cdr3Hasher, **kwargs) -> None:
        """
        Initiate Parametric Umap Transformer.

        Parameters
        ----------
        hasher : Cdr3Hasher
            Fitted hasher object.
        **kwargs
            Keyword arguments passed to `ParametricUMAP()` constructor.
        """
        self.hasher = hasher
        self.pumap = ParametricUMAP(
            verbose=True,
            n_training_epochs=kwargs.pop("n_training_epochs", 10),
            **kwargs
        )

    def __repr__(self) -> str:
        return "Parametric UMAP visualization"

    def fit(self, data_train: TcrCollection):
        """
        Train the UMAP using a train dataset.

        Parameters
        ----------
        data_train : TcrCollection
            TcrCollection of training data.
        """
        hashes = self.hasher.transform(data_train)
        self.pumap.fit(hashes)

    def transform(self, data: TcrCollection):
        """
        Use trained UMAP model to generate 2D coordinates from TCR sequences.

        Parameters
        ----------
        data : TcrCollection
            TcrCollection of training data.

        Returns
        -------
        pd.DataFrame
            Data with "x" and "y" fields representing UMAP coordinates.
        """
        hashes = self.hasher.transform(data)
        embedding = self.pumap.transform(hashes)
        data.data["x"], data.data["y"] = embedding.T
        return data.to_df()

    def plot(self, data: TcrCollection, color_feature: pd.Series) -> np.ndarray:
        pass

    @classmethod
    def from_file(cls, filepath: str):
        """
        Read a (trained) Parametric UMAP transformer from a local savefile.

        Parameters
        ----------
        filepath : str
            Path and name of folder where model is stored.
        """
        filepath = Path(filepath)
        pumap = load_ParametricUMAP(filepath / "ParametricUMAP")
        hasher = joblib.load(filepath / "Cdr3Hasher.joblib")
        res = cls(hasher)
        res.pumap = pumap
        return res

    def save(self, filepath: str):
        """
        Save a (trained) Parametric UMAP transformer to a local file.

        Parameters
        ----------
        filepath : str
            Filepath and name of folder to save model in.
        """
        filepath = Path(filepath)
        self.pumap.save(filepath / "ParametricUMAP")
        joblib.dump(self.hasher, filename=filepath / "Cdr3Hasher.joblib")


class ParametricUmapPlotter:
    def __init__(self, data: pd.DataFrame, background: pd.DataFrame = None) -> None:
        self.df = data
        self.bg_df = background

    def plot(
        self,
        ax=plt.Axes,
        color_feature: str = None,
        norm: str = None,
        plot_bg: bool = False
    ) -> plt.Axes:

        if self.df[color_feature].dtype.name in ["category", "object"]:
            # categorical coloring
            mapper = {
                i: c
                for i, c in zip(self.df[color_feature].unique(), cc.glasbey_category10)
            }
            self.df = self.df.sort_values(by=color_feature)
            c = self.df[color_feature].map(mapper).to_list()

            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=v,
                    label=k,
                    markersize=8,
                )
                for k, v in mapper.items()
            ]
            ax.legend(
                title=color_feature,
                handles=handles,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )

        else:
            # scalar color
            vmin = min(self.df[color_feature])
            vmax = max(self.df[color_feature])
            match norm:
                case "log": norm_ = LogNorm(vmin=vmin, vmax=vmax)
                case "symlog": norm_ = SymLogNorm(linthresh=0.3, linscale=0.3, vmin=vmin, vmax=vmax, base=10)
                case "symlog2": norm_ = SymLogNorm(linthresh=0.3, linscale=0.3, vmin=vmin, vmax=vmax, base=2)
                case _: norm_ = Normalize(vmin=vmin, vmax=vmax)

            match norm:
                case ("symlog" | "symlog2"): cmap_ = "RdBu_r"
                case _: cmap_ = sns.color_palette("rocket_r", as_cmap=True)

            sm = ScalarMappable(norm=norm_, cmap=cmap_)

            c = [sm.to_rgba(x) for x in self.df[color_feature]]

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax.get_figure().colorbar(sm, cax=cax, shrink=0.5)

        if plot_bg:
            ax.scatter(
                x=self.bg_df.x,
                y=self.bg_df.y,
                c='lightgrey',
                s=10,
                rasterized=True,
                alpha=0.4,
                linewidth=0,
            )


        ax.scatter(
            x=self.df.x,
            y=self.df.y,
            c=c,
            s=10,
            rasterized=True,
            alpha=0.4,
            linewidth=0,
        )

        sns.despine()

    def _relative_density(self, bw=None):
        emb_1 = self.df[["x", "y"]].T.to_numpy()
        emb_2 = self.bg_df[["x", "y"]].T.to_numpy()
        res = gaussian_kde(emb_1, bw_method=bw)(emb_1) / gaussian_kde(
            emb_2, bw_method=bw
        )(emb_1)
        return res

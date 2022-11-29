from itertools import repeat, chain
import joblib
from pathlib import Path

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


class ParametricUmapTransformer:
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

        if color_feature not in self.df.columns:
            self._parse_color_feature(color_feature)

        if self.df[color_feature].dtype.name in ["category", "object"]:
            # categorical coloring
            cmap = chain.from_iterable(repeat(cc.glasbey_category10))
            mapper = {
                i: c
                for i, c in zip(self.df[color_feature].unique(), cmap)
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

    def _parse_color_feature(self, color_feature) -> None:
        match color_feature.split('_'):
            case ["relative", "density", bw]:
                self.df[color_feature] = self._relative_density(bw)
            case ["clustcr", "cluster"]:
                self.df[color_feature] = self._clustcr_cluster()

    def _relative_density(self, bw=None) -> pd.Series:
        emb_1 = self.df[["x", "y"]].T.to_numpy()
        emb_2 = self.bg_df[["x", "y"]].T.to_numpy()
        res = gaussian_kde(emb_1, bw_method=bw)(emb_1) / gaussian_kde(
            emb_2, bw_method=bw
        )(emb_1)
        return res
    
    def _clustcr_cluster(self) -> pd.Series:

        try:
            from clustcr.clustering.clustering import Clustering
        except ImportError:
            raise ImportError('ClusTCR is not installed in current environment')

        cr = Clustering(method="mcl").fit(self.df["junction_aa"])

        df_ = pd.merge(
            how="left",
            left=self.df,
            right=cr.clusters_df,
            on="junction_aa",
        )

        return df_["cluster"].fillna(-1).astype(str).to_list()


from itertools import repeat, chain
import joblib
from pathlib import Path

import colorcet as cc
import numpy as np
import pandas as pd
from umap import ParametricUMAP
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, FuncNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.stats.mstats import winsorize
from raptcr.hashing import Cdr3Hasher
from raptcr.analysis import TcrCollection


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
        self.pumap_kwargs = kwargs

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
        pumap = ParametricUMAP(
                verbose=True,
                n_training_epochs=self.pumap_kwargs.pop("n_training_epochs", 10),
                **self.pumap_kwargs
            )
        pumap.fit(hashes)
        self.umap_encoder = pumap.encoder
        del pumap 


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
        embedding = self.umap_encoder.predict(hashes, batch_size=1000, verbose=True)
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
        hasher = joblib.load(filepath / "Cdr3Hasher.joblib")
        encoder = tf.keras.models.load_model(filepath / "encoder", compile=False)
        res = cls(hasher)
        res.umap_encoder = encoder
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
        self.umap_encoder.save(filepath / "encoder")
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
        plot_bg: bool = False,
        plot_legend : bool = False,
        winsorized: bool= False,
        **kwargs
    ) -> plt.Axes:

        if color_feature not in self.df.columns:
            self._parse_color_feature(color_feature)

        if self.df[color_feature].dtype.name in ["category", "object"]:
            # categorical coloring
            colors = chain.from_iterable(repeat(cc.glasbey_category10))
            mapper = {
                i: c
                for i, c in zip(self.df[color_feature].unique(), colors)
            }
            if None in self.df[color_feature].to_list():
                mapper[None] = [0.7,0.7,0.7]

            if color_feature in ["clustcr_cluster"]:
                pass
            else:
                self.df = self.df.sort_values(by=color_feature)

            c = self.df[color_feature].map(mapper).to_list()

            if plot_legend:
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
            if not winsorized:
                vmin = min(self.df[color_feature].replace([np.inf, -np.inf],np.nan))
                vmax = max(self.df[color_feature].replace([np.inf, -np.inf],np.nan))
            else:
                vmin = min(winsorize(self.df['relative_density_0.05'], limits=(0.01,0.01)))
                vmax = max(winsorize(self.df['relative_density_0.05'], limits=(0.01,0.01)))

            if norm == "log":
                norm_ = LogNorm(vmin=vmin, vmax=vmax)
            elif norm == "log2":
                norm_ = FuncNorm(functions=(np.log2, lambda x: 2**x))
            elif norm == "symlog":
                norm_ = SymLogNorm(linthresh=0.3, linscale=0.3, vmin=vmin, vmax=vmax, base=10)
            elif norm == "symlog2":
                norm_ = SymLogNorm(linthresh=0.3, linscale=0.3, vmin=vmin, vmax=vmax, base=2)
            else:
                norm_ = Normalize(vmin=vmin, vmax=vmax)

            if norm in ["symlog", "symlog2"]:
                cmap_ = "RdBu_r"
            else:
                cmap_ = sns.color_palette("rocket_r", as_cmap=True) 

            sm = ScalarMappable(norm=norm_, cmap=cmap_)

            c = [sm.to_rgba(x) for x in self.df[color_feature]]

            if plot_legend:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                ax.get_figure().colorbar(sm, cax=cax, shrink=0.5)

        if plot_bg:
            ax.scatter(
                x=self.bg_df.x,
                y=self.bg_df.y,
                c='lightgrey',
                s=5,
                rasterized=True,
                alpha=0.5,
                linewidth=0,
                **kwargs
            )


        ax.scatter(
            x=self.df.x,
            y=self.df.y,
            c=c,
            s=5,
            rasterized=True,
            alpha=0.5,
            linewidth=0,
        )

        sns.despine()

    def _parse_color_feature(self, color_feature) -> None:
        """
        Parses input color feature, computes and adds column to self.df for
        specific features where a function is available.
        """
        return NotImplementedError

    def _relative_density(self, bw=None) -> pd.Series:
        """
        Compare the density of two scatterplots in each point using kernel
        density estimates.

        Parameters
        ----------
        bw : float, optional
            Kernel density estimation bandwidth.
        """
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

        res = df_["cluster"]
        res = np.where(pd.isna(res), None, res.fillna(-1).astype(int).astype(str))
        return res

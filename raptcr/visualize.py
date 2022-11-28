from abc import ABC, abstractmethod
import joblib
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from umap import ParametricUMAP
from umap.parametric_umap import load_ParametricUMAP
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm, LogNorm, FuncNorm, Normalize
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


class ParametricUmap(BaseVisualizer):
    def __init__(self, hasher: Cdr3Hasher, **kwargs) -> None:
        """
        Initiate ParametricUmap Transformer.

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
        size_feature: str = None,
        size_norm: str = None,
        hue_norm: str = None,
        **kwargs
    ) -> plt.Axes:

        size_norm = self._parse_norms(size_norm)
        hue_norm = self._parse_norms(size_norm)
        color_feature = self._parse_color_feature(color_feature)

        if self.bg_df is not None:
            sns.scatterplot(
                ax=ax,
                data=self.bg_df,
                x="x",
                y="y",
                size=size_feature,
                size_norm=size_norm,
                s=2,
                color="lightgrey",
                linewidth=0,
                alpha=0.4,
                rasterized=True,
            )

        sns.scatterplot(
            ax=ax,
            data=self.df.sort_values(color_feature),
            x="x",
            y="y",
            size=size_feature,
            size_norm=size_norm,
            sizes=(1, 15),
            hue=color_feature,
            hue_norm=hue_norm,
            linewidth=0,
            alpha=0.4,
            rasterized=True,
            palette=kwargs.get("palette", "rocket_r"),
        )

        # place legend outside
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

        return ax

    def _parse_norms(self, norm: str) -> Normalize:
        norm_map = {
            "log": LogNorm(),
            "log2": FuncNorm(functions=(np.log2, lambda x: 2**x)),
            None: NoNorm(),
        }

        return norm_map[norm]

    def _parse_color_feature(self, color_feature) -> Union[str, pd.Series]:
        if color_feature in self.df:
            return color_feature

        match color_feature.split("_"):
            case ["relative", "density"]:
                return self._relative_density()
            case ["relative", "density", bw]:
                return self._relative_density(bw=bw)

    def _relative_density(self, bw=None):
        emb_1 = self.df[["x", "y"]].T.to_numpy()
        emb_2 = self.bg_df[["x", "y"]].sample(len(self.df)).T.to_numpy()
        res = gaussian_kde(emb_1, bw_method=bw)(emb_1) / gaussian_kde(
            emb_2, bw_method=bw
        )(emb_1)
        return pd.Series(res)

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
        size_feature: str = None,
        **kwargs
    ) -> plt.Axes:

        if color_feature:
            color_feature = self._parse_color_feature(color_feature)
            self.df = self.df.sort_values(color_feature)
        size_feature_kwargs = self._parse_size_feature(size_feature)

        if self.bg_df is not None:
            sns.scatterplot(
                ax=ax,
                data=self.bg_df,
                x="x",
                y="y",
                color="lightgrey",
                linewidth=0,
                alpha=0.4,
                rasterized=True,
                s = 3
            )

        sns.scatterplot(
            ax=ax,
            data=self.df,
            x="x",
            y="y",
            hue=color_feature,
            linewidth=0,
            alpha=0.4,
            rasterized=True,
            cmap="rocket_r",
            **size_feature_kwargs,
        )

        # place legend outside
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

        return ax

    def _parse_color_feature(self, color_feature) -> Union[str, pd.Series]:
        if color_feature in self.df:
            return color_feature

        match color_feature.split("_"):
            case ["relative", "density"]:
                self.df[color_feature] = self._relative_density()
            case ["relative", "density", bw]:
                self.df[color_feature] = self._relative_density(bw=float(bw))

        return color_feature

    def _parse_size_feature(self, size_feature) -> Union[str, pd.Series]:
        if not size_feature:
            return dict(s=3)

        if size_feature in self.df:
            return dict(size=size_feature, sizes=(1,30))

    def _relative_density(self, bw=None):
        emb_1 = self.df[["x", "y"]].T.to_numpy()
        emb_2 = self.bg_df[["x", "y"]].T.to_numpy()
        res = gaussian_kde(emb_1, bw_method=bw)(emb_1) / gaussian_kde(
            emb_2, bw_method=bw
        )(emb_1)
        return res


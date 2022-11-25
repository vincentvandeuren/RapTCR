from abc import ABC, abstractmethod
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from umap import ParametricUMAP
from umap.parametric_umap import load_ParametricUMAP
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
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


class ParametricUmapVisualizer(BaseVisualizer):
    def __init__(self, hasher: Cdr3Hasher, **kwargs) -> None:
        """
        Initiate ParametricUmapVisualizer.

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

        df = self.transform(data)

        fig, ax = plt.subplots(figsize=(8, 8))

        return sns.scatterplot(
            data=df,
            x="x",
            y="y",
            s=0.5,
            hue=color_feature,
            palette="rocket_r",
            rasterized=True,
            ax=ax,
        )

    @classmethod
    def from_file(cls, filepath: str):
        """
        Read a (trained) ParametricUmapVisualizer from a local savefile.

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
        Save a (trained) ParametricUmapVisualizer to a local file.

        Parameters
        ----------
        filepath : str
            Filepath and name of folder to save model in.
        """
        filepath = Path(filepath)
        self.pumap.save(filepath / "ParametricUMAP")
        joblib.dump(self.hasher, filename=filepath / "Cdr3Hasher.joblib")

from ..hashing import Cdr3Hasher
from ..analysis import TcrCollection

import pandas as pd
from umap import UMAP
from sklearn.base import TransformerMixin

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import plotly.express as px
import natsort

class UmapTransformer(TransformerMixin):
    def __init__(self, hasher: Cdr3Hasher, umap:UMAP=None) -> None:
        """
        Initiate Umap Transformer.

        Parameters
        ----------
        hasher : Cdr3Hasher
            Fitted hasher object.
        umap : UMAP
            Fitted umap transformer.
        **kwargs
            Keyword arguments passed to `ParametricUMAP()` constructor.
        """
        if not umap:
            umap = UMAP(
                n_neighbors=6,
                min_dist=0.01,
                negative_sample_rate=30,
                local_connectivity=hasher.m
                )
        self.hasher = hasher
        self.umap = umap

    def __repr__(self) -> str:
        return "UMAP transformer"

    def fit(self, data_train: TcrCollection):
        """
        Fit UMAP.

        Parameters
        ----------
        data_train : TcrCollection
            TcrCollection of training data.
        """
        hashes = self.hasher.transform(data_train)
        self.umap.fit(hashes)
        return self

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
        embedding = self.umap.transform(hashes)
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
        raise NotImplementedError()

    def save(self, filepath: str):
        """
        Save a (trained) Parametric UMAP transformer to a local file.

        Parameters
        ----------
        filepath : str
            Filepath and name of folder to save model in.
        """
        raise NotImplementedError()


def plot_umap(df:pd.DataFrame, ax:Axes=None, hue:str=None, **kwargs) -> Axes:
    """
    Plot the results of a UMAP transformation.

    Parameters
    ----------
    df : pd.DataFrame
        The output dataframe of the UmapTransformer. Columns can be modified.   
    hue : str
        Column name of the hue feature.
    **kwargs
        Other keyword params passed to a sns.scatterplot function, i.e. size, color, alpha.
    """
    ax = ax or plt.gca()
    alpha = kwargs.pop("alpha", 0.2)
    s = kwargs.pop("s", 8)

    # plot
    sns.scatterplot(
        data = df,
        x="x",
        y="y",
        s = s,
        linewidth=0,
        ax = ax,
        alpha=alpha,
        hue = hue,
        **kwargs
    )
    plt.legend(bbox_to_anchor=(1,1), loc="upper left") #add name
    sns.despine()
    return ax


def plot_interactive_umap(filename:str, df, color, **kwargs) -> None:
    """
    Plot an interactive scatterplot of the UMAP transformation using plotly. 
    The resulting html can be opened in a browser.

    df : pd.DataFrame
        The output dataframe of the UmapTransformer. Columns can be modified.   
    filename : str
        Filepath where output html plot is saved.
    color : str
        Column name of the hue feature.
    **kwargs
        Other keyword params passed to a px.scatter function, i.e. size, transparency.
    """

    fig = px.scatter(
        df.sort_values(by=color, key=natsort.natsort_keygen(alg=natsort.ns.REAL)), 
        x="x", 
        y="y", 
        color = color,
        hover_name="junction_aa", 
        hover_data=df.columns.to_list(), 
        **kwargs
        )

    fig.write_html(filename)
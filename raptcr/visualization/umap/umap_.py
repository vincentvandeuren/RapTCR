from ...hashing import Cdr3Embedder
from ...analysis import Repertoire

import pandas as pd
from umap.umap_ import UMAP
from sklearn.base import TransformerMixin

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import plotly.express as px
import natsort

class UmapTransformer(TransformerMixin):
    def __init__(self, embedder: Cdr3Embedder, umap:UMAP=None, **kwargs) -> None:
        """
        Initiate Umap Transformer.

        Parameters
        ----------
        embedder : Cdf rCdr3Embedder
            Fitted embedder object.
        umap : UMAP
            Fitted umap transformer.
        """
        if not umap:
            n_neighbors = kwargs.pop("n_neighbors", 6)
            min_dist = kwargs.pop("min_dist", 0)
            negative_sample_rate = kwargs.pop("negative_sample_rate", 30)
            local_connectivity = kwargs.pop("local_connectivity", embedder.m)

            umap = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                negative_sample_rate=negative_sample_rate,
                local_connectivity=local_connectivity,
                **kwargs
                )
        self.embedder = embedder
        self.umap = umap

    def __repr__(self) -> str:
        return "UMAP transformer"

    def fit(self, data_train: Repertoire):
        """
        Fit UMAP.

        Parameters
        ----------
        data_train : Repertoire
            Repertoire of training data.
        """
        hashes = self.embedder.transform(data_train)
        self.umap.fit(hashes)
        return self

    def transform(self, data: Repertoire):
        """
        Use trained UMAP model to generate 2D coordinates from TCR sequences.

        Parameters
        ----------
        data : Repertoire
            Repertoire of training data.

        Returns
        -------
        pd.DataFrame
            Data with "x" and "y" fields representing UMAP coordinates.
        """
        hashes = self.embedder.transform(data)
        embedding = self.umap.transform(hashes)
        data["x"], data["y"] = embedding.T
        return data

    @classmethod
    def from_file(cls, filepath: str):
        """
        Read a (trained) UMAP transformer from a local savefile.

        Parameters
        ----------
        filepath : str
            Path and name of folder where model is stored.
        """
        raise NotImplementedError()

    def save(self, filepath: str):
        """
        Save a (trained) UMAP transformer to a local file.

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


def plot_interactive_umap(filename:str, df, color=None, **kwargs) -> None:
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
    if not filename.endswith(".html"):
        filename += ".html"

    if color:
        kwargs["color"] = color
        df = df.sort_values(by=color, key=natsort.natsort_keygen(alg=natsort.ns.REAL))

    fig = px.scatter(
        df, 
        x="x", 
        y="y", 
        hover_name="junction_aa", 
        hover_data=df.columns.to_list(), 
        **kwargs
        )

    fig.write_html(filename)
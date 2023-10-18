import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import plotly.express as px
import natsort

from ..analysis import Repertoire

def _assert_positional_information_present(rep:Repertoire):
    if not {"x", "y"}.issubset(rep.columns):
        raise ValueError(
            """x and y columns are not defined, run your Repertoire through a 
            dimensionality reducer (e.g. UmapTransformer) first"""
            )

def plot_interactive(filename:str, rep, color=None, **kwargs) -> None:
    """
    Plot an interactive scatterplot of the UMAP transformation using plotly. 
    The resulting html can be opened in a browser.

    rep : pd.DataFrame
        The output dataframe of the UmapTransformer. Columns can be modified.   
    filename : str
        Filepath where output html plot is saved.
    color : str
        Column name of the hue feature.
    **kwargs
        Other keyword params passed to a px.scatter function, i.e. size, transparency.
    """
    _assert_positional_information_present(rep)

    if not filename.endswith(".html"):
        filename += ".html"

    if color:
        kwargs["color"] = color
        rep = rep.sort_values(by=color, key=natsort.natsort_keygen(alg=natsort.ns.REAL))

    fig = px.scatter(
        rep, 
        x="x", 
        y="y", 
        hover_name="junction_aa", 
        hover_data=rep.columns.to_list(), 
        **kwargs
        )

    fig.write_html(filename)

def plot(rep:Repertoire, ax:Axes=None, hue:str=None, **kwargs) -> Axes:
    """
    Plot the results of a UMAP transformation.

    Parameters
    ----------
    rep : pd.DataFrame
        The output dataframe of the UmapTransformer. Columns can be modified.   
    hue : str
        Column name of the hue feature.
    **kwargs
        Other keyword params passed to a sns.scatterplot function, i.e. size, color, alpha.
    """
    _assert_positional_information_present(rep)

    ax = ax or plt.gca()
    alpha = kwargs.pop("alpha", 0.2)
    s = kwargs.pop("s", 8)

    # plot
    sns.scatterplot(
        data = rep,
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
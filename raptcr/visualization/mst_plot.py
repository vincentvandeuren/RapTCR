from ..indexing import BaseIndex 
from ..analysis import ClusteredRepertoire

from itertools import repeat
from typing import List, Tuple

from sklearn.base import TransformerMixin

import tmap as tm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import plotly.express as px

class MstTransformer(TransformerMixin):
    def __init__(self, index:BaseIndex) -> None:
        """Initiate MST transformer"""  
        self.idx = index
        self.tmap_config = tm.LayoutConfiguration()


    def fit(self, X:ClusteredRepertoire=None):
        return self

    def transform(self, X:ClusteredRepertoire, y=None):
        cluster_id_list = {i:c for i,c in enumerate([c for c in X.iter_clusters()])}
        vertex_count = len(cluster_id_list)
        clusters = list(cluster_id_list.values())
        idx = self.idx.add(clusters)
        knnr = idx.knn_search() #todo make work with other indexes
        edge_list = self._knnr_to_edgelist(knnr)

        x, y, s, t, _ = tm.layout_from_edge_list_native(
            vertex_count=vertex_count,
            edges=edge_list,
            config=self.tmap_config,
        )

        return MstResult(X, x,y,s,t,cluster_id_list)


    def _knnr_to_edgelist(self, knnr) -> List[Tuple[int, int, float]]:
        edge_list = [
            a for b in [
                tuple(zip(repeat(i_1), i_n, score))
                for i_1, i_n, score in zip(range(len(knnr.I)), knnr.I, knnr.D)
            ] for a in b if a[0] != a[1]
            ]

        return edge_list


class MstResult:
    def __init__(self,crep,x,y,s,t,cluster_ids) -> None:
        self.crep = crep
        self.x = x
        self.y = y
        self.s = s
        self.t = t
        self.cluster_ids = cluster_ids


    def sequences_df(self) -> pd.DataFrame:
        mst_df = pd.DataFrame({
            "cluster":[str(x) for x in self.cluster_ids.keys()],
            "x":self.x,
            "y":self.y,
        }).set_index("cluster")
        return pd.merge(right=mst_df, left=self.crep.data, on="cluster")

    def clusters_df(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "cluster" : self.cluster_ids.keys(),
            "motif" : [c.motif() for c in self.crep.iter_clusters()],
            "size" : [len(c) for c in self.crep.iter_clusters()],
            "x":self.x,
            "y":self.y
        }).set_index("cluster")
        return df


def plot_mst(mst_result:MstResult, clusters_df=None, ax:Axes=None, edge_kwargs:dict=dict(), node_kwargs:dict=dict()) -> Axes:
    clusters_df = clusters_df or mst_result.clusters_df()
    ax = ax or plt.gca()
    coords = {i: v for i, v in enumerate(zip(mst_result.x, mst_result.y))}
    # plot edges
    edge_color = edge_kwargs.pop('color', "#7F7F7F")
    edge_linewidth = edge_kwargs.pop('linewidth', 0.5)

    for a, b in list(zip(mst_result.s, mst_result.t)):
        a, b = coords[a], coords[b]
        xc, yc = zip(a, b)
        ax.plot(xc, yc, color=edge_color, linewidth=edge_linewidth, zorder=0, **edge_kwargs)

    # plot nodes
    node_size = node_kwargs.pop("size", "size")
    sns.scatterplot(
        data = mst_result.clusters_df(),
        x = "x",
        y = "y",
        size = node_size,
        linewidth = 0,
        ax=ax,
        **node_kwargs
    )
    return ax


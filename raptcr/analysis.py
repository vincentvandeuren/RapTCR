import pandas as pd
from abc import ABC
from functools import cached_property

from .tools import profile_matrix, motif_from_profile

class TcrCollection(ABC):
    def __init__(self, df) -> None:
        # assert all required columns are present
        required_cols = self.required_cols if hasattr(self,"required_cols") else ["v_call", "j_call", "junction_aa"]
        if not set(required_cols).issubset(df.columns):
            raise ValueError(
                f"Input dataframe requires following columns: {required_cols}"
            )

        self.data = df

    def __repr__(self) -> str:
        return f"TCR collection of size {len(self.data)}"

    def __iter__(self):
        for s in self.cdr3s:
            yield s

    def __getitem__(self, n):
        return self.cdr3s[n]

    def __len__(self):
        return len(self.data)

    def to_df(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        return self.data

    def sample(self, n:int, weight_col:str=None):
        """
        Randomly sample n TCRs.

        Parameters
        ----------
        n : int
            Number of TCRs to sample.
        weight_col : str or array-like, optional
            If not passed, each TCR has equal probability to be picked. If an
            array is passed, array values determine the probabilty of TCR at
            that index to be picked. If a string is passed, the corresponding
            column from self.data will be used as weight.
        """
        return self.__class__(df=self.data.sample(n, weights=weight_col))

    @cached_property
    def cdr3s(self):
        return self.data["junction_aa"].to_list()


class Repertoire(TcrCollection):
    """Class for storing TCR-seq information and calculating useful properties."""

    def __repr__(self) -> str:
        return f"TCR repertoire of size {len(self.data)}"


class Cluster(TcrCollection):
    """
    Class for storing Cluster-level clonotype information, and calculating its
    useful properties.
    """

    def __repr__(self) -> str:
        return f"TCR cluster of size {len(self.data)}"


    def motif(self, method="standard", cutoff=0.7) -> str:
        return motif_from_profile(profile_matrix(self.cdr3s), method, cutoff)


class ClusteredRepertoire(TcrCollection):
    """
    Class for storing clustered clonotype information.
    """
    def __init__(self, df) -> None:
        self.cluster_ids = df.index.get_level_values(level="cluster").unique().to_list()
        super().__init__(df)

    def __repr__(self) -> str:
        if not "-1" in self.cluster_ids:
            return f"Clustered TCR repertoire of size {len(self.data)} containing {len(self.cluster_ids)} clusters."
        else:
            num_unclustered_sequences = len(self.data.xs("-1", level="cluster"))
            return f"Clustered TCR repertoire of size {len(self.data)} containing {len(self.cluster_ids)-1} clusters and {num_unclustered_sequences} unclustered sequences."

    @classmethod
    def from_clustcr_result(cls, df, clusteringresult):
        df_merged = pd.merge(left=df, right=clusteringresult.clusters_df, how="left")
        df_merged["cluster"] = df_merged["cluster"].fillna(-1).astype(int).astype(str)
        df_merged=df_merged.reset_index().set_index(["cluster", "index"])
        return cls(df_merged)

    def __iter__(self):
        for i in self.cluster_ids:
            if i == "-1":
                yield Repertoire(self.data.xs(i, level=0))
            else:
                yield Cluster(self.data.xs(i, level=0))

    def iter_clusters(self):
        for i in self.cluster_ids:
            if i != "-1":
                yield Cluster(self.data.xs(i, level=0))

    def __len__(self):
        return len(self.data)
import pandas as pd
from abc import ABC
from functools import cached_property


class TcrCollection(ABC):
    def __init__(self, df) -> None:
        # assert all required columns are present
        required_cols = ["v_call", "j_call", "junction_aa"]
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

    def to_df(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        return self.data

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

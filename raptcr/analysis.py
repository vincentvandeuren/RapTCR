import pandas as pd
from abc import ABC

class TcrCollection(ABC):
    def __init__(self, df) -> None:
        # assert all required columns are present
        required_cols = ["v_call", "j_call", "junction_aa"]
        if not set(required_cols).issubset(df.columns):
            raise ValueError(f"Input dataframe requires following columns: {required_cols}")

        self.data = df

    def __repr__(self) -> str:
        return f"TCR collection of size {len(self.data)}"

    def to_df(self) -> pd.DataFrame:
        """Convert to pandas DataFrame
        """
        return self.data


class Repertoire(TcrCollection):
    """Class for storing TCR-seq information and calculating useful properties.
    """
    def __repr__(self) -> str:
        return f"TCR repertoire of size {len(self.data)}"

class Cluster(TcrCollection):
    """
    Class for storing Cluster-level clonotype information, and calculating its
    useful properties.
    """
    def __repr__(self) -> str:
        return f"TCR cluster of size {len(self.data)}"
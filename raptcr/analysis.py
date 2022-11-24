from itertools import count
from typing import Iterable, List
import pandas as pd

class Repertoire():
    """
    Class for storing TCR-seq information and calculating useful properties.
    """
    def __init__(self, df) -> None:
        self.data = df

    def __repr__(self) -> str:
        return f"TCR repertoire of size {len(self.data)}"

    def to_df(self) -> pd.DataFrame:
        """Convert to pandas DataFrame
        """
        return self.data
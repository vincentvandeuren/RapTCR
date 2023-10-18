import pandas as pd
import warnings


from .tools import profile_matrix, motif_from_profile

class Repertoire(pd.DataFrame):
    # Repertoire-level class attributes here:
    _metadata = ['repertoire_id']

    @property
    def _constructor(self):
        """This is the key to letting Pandas know how to keep
        derivative `SomeData` the same type as yours.  It should
        be enough to return the name of the Class.  However, in
        some cases, `__finalize__` is not called and `my_attr` is
        not carried over.  We can fix that by constructing a callable
        that makes sure to call `__finalize__` every time."""
        def _c(*args, **kwargs):
            return Repertoire(*args, **kwargs).__finalize__(self)
        return _c

    def __init__(self, *args, **kwargs):
        # grab and set the repertoire-level attributes from the keyword arguments
        self.repertoire_id = kwargs.pop("repertoire_id", None)
        super().__init__(*args, **kwargs) # initiate df
        self._validate()


    def _validate(self):
        # assert all required columns are present
        required_cols = ["v_call", "j_call", "junction_aa"]
        if not set(required_cols).issubset(self.columns):
            raise ValueError(
                f"Input dataframe requires following columns: {required_cols}"
            )

        # add sequence_id column if not present
        if "sequence_id" not in self.columns:
            self["sequence_id"] = [str(i) for i in range(len(self))]

        # assert all sequence_ids are unique
        if not self["sequence_id"].is_unique:
            warnings.warn("Not all sequence ids are unique", stacklevel=2)

    def to_imgt(self):
        """
        Parse V and J gene columns to IMGT format.
        """
        raise NotImplementedError


    def sample(self, weight_col:str="duplicate_count", **kwargs):
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
        if weight_col not in self.columns:
            warnings.warn(
                f"`{weight_col}` is not present in your data, sampling was performed without taking frequency information into account.",
                stacklevel=2
            )
            weight_col = None
        return super().sample(weights=weight_col, **kwargs)

    def create_motif(self, method="standard", cutoff=0.7):
        seqs = self.cdr3s
        if len(set(map(len, seqs))) > 1: # seqs of different lengths
            warnings.warn('not all sequences have the same lenght, only those with the most common length get retained in the motif', stacklevel=2)
        return motif_from_profile(profile_matrix(self.cdr3s), method, cutoff)

    @property
    def cdr3s(self):
        return self.junction_aa.to_numpy()

    @property
    def data(self):
        """
        temporary attribute as to not break methods that used the older Repertoire object
        """
        return self

    def _iter_sequences(self):
        for s in self.junction_aa:
            yield s

    def __repr__(self) -> str:
        # workaround: do not validate df when printing
        return pd.DataFrame(self).__repr__()
    
    def _repr_html_(self):
        # workaround: do not validate df when printing
        return pd.DataFrame(self)._repr_html_()

#TODO recreate ClusteredRepertoire subclass of Repertoire. Require cluster_id
#column (str, unclustered="-1") and useful methods for cluster analysis.

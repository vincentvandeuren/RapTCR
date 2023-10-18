from ...hashing import Cdr3Embedder
from ...analysis import Repertoire

from umap.umap_ import UMAP
from sklearn.base import TransformerMixin


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






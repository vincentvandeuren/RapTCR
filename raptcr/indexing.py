from abc import ABC
from functools import partial
from typing import List, Tuple

import faiss
from pynndescent import NNDescent
import numpy as np

from .hashing import Cdr3Hasher
from .analysis import TcrCollection


class BaseIndex(ABC):
    """
    Abstract structure for an index, supports adding CDR3s and searching them.
    """

    def __init__(self, idx: faiss.Index, hasher: Cdr3Hasher) -> None:
        super().__init__()
        self.idx = idx
        self.hasher = hasher
        self.ids = {}

    def _add_hashes(self, hashes):
        if not self.idx.is_trained:
            self.idx.train(hashes)
        self.idx.add(hashes)

    def _add_ids(self, X):
        for i, x in enumerate(X):
            self.ids[i] = x

    def add(self, X: TcrCollection):
        """
        Add sequences to the index.

        Parameters
        ----------
        X : TcrCollection
            Collection of TCRs to add. Can be a Repertoire, list of Clusters, or
            list of str.
        """
        hashes = self.hasher.transform(X).astype(np.float32)
        self._add_hashes(hashes)
        self._add_ids(X)

    def _assert_trained(self):
        if not self.idx.is_trained:
            raise ValueError("Index is untrained, please add first.")

    def _search(self, x, k):
        return self.idx.search(x=x, k=k)

    def knn_search(self, y: TcrCollection, k: int = 100):
        """
        Search index for k-nearest neighbours.

        Parameters
        ----------
        y : TcrCollection
            Query TCRs.
        k : int, default = 100
            Number of nearest neighbours to search.

        Returns
        -------
        KnnResult
            `KnnResult` object.
        """
        self._assert_trained()
        hashes = self.hasher.transform(y).astype(np.float32)
        D, I = self._search(x=hashes, k=k)
        return KnnResult(y, D, I, self.ids)


class ExactIndex(BaseIndex):
    """
    Exact search for euclidean hash distance.
    """

    def __init__(self, hasher: Cdr3Hasher) -> None:
        """
        Initialize index.

        Parameters
        ----------
        hasher : Cdr3Hasher
            Fitted Cdr3Hasher class.
        """
        idx = faiss.IndexFlatL2(64)
        super().__init__(idx, hasher)


class PynndescentIndex(BaseIndex):
    """
    Approximate search using PyNNDescent.
    """

    def __init__(
        self,
        hasher: Cdr3Hasher,
        n_neighbors: int = 100,
        diversify_prob: float = 1.0,
        pruning_degree_multiplier: float = 1.5,
    ):
        """
        Initialize index.
        """
        idx = partial(
            NNDescent,
            n_neighbors=n_neighbors,
            diversify_prob=diversify_prob,
            pruning_degree_multiplier=pruning_degree_multiplier,
        )
        idx.is_trained = True
        super().__init__(idx, hasher)


    def _add_hashes(self, X):
        self.idx = self.idx(X)
        self.idx.is_trained = True
    
    def _search(self, x, k=None):
        return self.idx.query(x)

    def knn_search(self, y: TcrCollection=None):
        """
        Search index for nearest neighbours.

        Parameters
        ----------
        y : TcrCollection, optional
            Pass query TCRs if subset is needed. If not passed, returns the
            neighbours used when adding the data to the index, which is much
            faster.
        """
        if not y:
            return self.idx.neighbor_graph
        return super().knn_search(y)
    

class BaseApproximateIndex(BaseIndex):
    """
    Abstract class for approximate indexes implementing the `n_probe` property.
    """

    @property
    def n_probe(self):
        return faiss.extract_index_ivf(self.idx).nprobe

    @n_probe.setter
    def n_probe(self, n: int):
        ivf = faiss.extract_index_ivf(self.idx)
        ivf.nprobe = n


class IvfIndex(BaseApproximateIndex):
    def __init__(
        self, hasher: Cdr3Hasher, n_centroids: int = 32, n_probe: int = 5
    ) -> None:
        """
        Inverted file index for approximate nearest neighbour search.

        Parameters
        ----------
        hasher : Cdr3Hasher
            Fitted hasher object to transform CDR3 to vectors.
        n_centroids : int, default=32
            Number of centroids for the initial k-means clustering.
        n_probe : int, default=5
            Number of centroids to search at query time. Higher n_probe means
            higher recall, but slower speed.
        """
        idx = faiss.index_factory(64, f"IVF{n_centroids},Flat")
        super().__init__(idx, hasher)
        self.n_probe = n_probe


class HnswIndex:
    def __init__(self, hasher: Cdr3Hasher, n_links: int = 32) -> None:
        """
        Index based on Hierarchical Navigable Small World networks.

        Parameters
        ----------
        hasher : Cdr3Hasher
            Fitted hasher object to transform CDR3 to vectors.
        n_links : int, default=32
            Number of bi-directional links created for each element during index
            construction. Increasing M leads to better recall but higher memory
            size and slightly slower searching.

        """
        idx = faiss.index_factory(64, f"HNSW{n_links},Flat")
        super().__init__(idx, hasher)


class FastApproximateIndex(BaseApproximateIndex):
    def __init__(
        self,
        hasher: Cdr3Hasher,
        n_centroids: int = 256,
        n_links: int = 32,
        n_probe: int = 10,
    ) -> None:
        """
        Approximate index based on a combination of IVF and HNSW using scalar
        quantizer encoding.

        Parameters
        ----------

        hasher : Cdr3Hasher
            Fitted hasher object to transform CDR3 to vectors.
        n_centroids : int, default=32
            Number of centroids for the initial k-means clustering.
        n_probe : int, default=5
            Number of centroids to search at query time. Higher n_probe means
            higher recall, but slower speed.
        n_links : int, default=32
            Number of bi-directional links created for each element during index
            construction. Increasing M leads to better recall but higher memory
            size and slightly slower searching.
        """
        idx = faiss.index_factory(64, f"IVF{n_centroids}_HNSW{n_links},SQ6")
        super().__init__(idx, hasher)
        self.n_probe = n_probe


class KnnResult:
    """
    Result of k-nearest neighbor search.
    """

    def __init__(self, y, D, I, ids) -> None:
        self.D = np.sqrt(D)
        self.I = I
        self.y_idx = {y: i for i, y in enumerate(y)}
        self.ids = ids

    def __repr__(self) -> str:
        s, k = self.D.shape
        return f"k-nearest neighbours result (k={k}, size={s})"

    def extract_neighbours(self, cdr3: str) -> List[Tuple[str, float]]:
        """
        Query the KnnResult for neighbours of a specific sequence.

        Parameters:
        -----------
        cdr3 : str
            Query sequence.

        Returns
        -------
        List[(str, float)]
            List of matches, containing (sequence, score) tuples.
        """
        try:
            i = self.y_idx[cdr3]
        except KeyError:
            raise KeyError(f"{cdr3} was not part of your query")
        I_ = np.vectorize(self._annotate_id)(self.I[i])
        return list(zip(I_, self.D[i]))

    def _annotate_id(self, cdr3_id):
        return self.ids.get(cdr3_id)

    def as_network(self, max_edge: float = 15):
        raise NotImplementedError()

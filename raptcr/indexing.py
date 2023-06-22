from abc import ABC
from functools import partial, cached_property
from typing import List, Tuple, Callable

import faiss
import numpy as np
import pandas as pd

from raptcr.analysis import Repertoire

from .hashing import Cdr3Embedder
from .analysis import Repertoire


class BaseIndex(ABC):
    """
    Abstract structure for an index, supports adding CDR3s and searching them.
    """

    def __init__(self, idx: faiss.Index, embedder: Cdr3Embedder) -> None:
        super().__init__()
        self.idx = idx
        self.embedder = embedder
        self.ids = np.array([])
        self.rep = None

    def _add_hashes(self, hashes):
        if not self.idx.is_trained:
            self.idx.train(hashes)
        self.idx.add(hashes)

    def _add_ids(self, X):
        ids = X.cdr3s if isinstance(X, Repertoire) else list(X)
        self.ids = np.hstack((self.ids, ids))

    def _add_repertoire(self, X):
        if self.rep is None:
            self.rep = X
        else:
            self.rep = pd.concat([self.rep, X])

    def add(self, X: Repertoire):
        """
        Add sequences to the index.

        Parameters
        ----------
        X : Repertoire
            Collection of TCRs to add. Can be a Repertoire, list of Clusters, or
            list of str.
        """
        hashes = self.embedder.transform(X).astype(np.float32)
        self._add_hashes(hashes)
        self._add_ids(X)
        if isinstance(X, Repertoire):
            self._add_repertoire(X)
        return self

    def _assert_trained(self):
        if not self.idx.is_trained:
            raise ValueError("Index is untrained, please add first.")

    def _search(self, x, k):
        return self.idx.search(x=x, k=k)

    def knn_search(self, y: Repertoire, k: int = 100):
        """
        Search index for k-nearest neighbours.

        Parameters
        ----------
        y : Repertoire
            Query TCRs.
        k : int, default = 100
            Number of nearest neighbours to search.

        Returns
        -------
        KnnResult
            `KnnResult` object.
        """
        self._assert_trained()
        hashes = self.embedder.transform(y).astype(np.float32)
        D, I = self._search(x=hashes, k=k)
        return KnnResult(y, D, I, self.ids)
    
    def _radius_search(self, x, d):
        radius = d**2
        lims, D, I = self.idx.range_search(x, radius)
        return lims, np.sqrt(D), I
        
    def radius_search(self, query:Repertoire, dist:float) -> pd.DataFrame:
        hashes = self.embedder.transform(query)

        lims, D, I = self._radius_search(hashes, dist)

        return RadiusSearchResult(lims, D, I, query, self.ids, dist, self.rep)


class FlatIndex(BaseIndex):
    """
    Exact search for euclidean hash distance.
    """

    def __init__(self, embedder: Cdr3Embedder) -> None:
        """
        Initialize index.

        Parameters
        ----------
        embedder : Cdr3Embedder
            Fitted Cdr3Embedder class.
        """
        idx = faiss.IndexFlatL2(embedder.m)
        super().__init__(idx, embedder)

class FlatGpuIndex(BaseIndex):
    """
    Exact search for euclidean hash distance.
    """

    def __init__(self, embedder: Cdr3Embedder) -> None:
        """
        Initialize index.

        Parameters
        ----------
        embedder : Cdr3Embedder
            Fitted Cdr3Embedder class.
        """
        idx_cpu = faiss.IndexFlatL2(embedder.m)
        try:
            self.resource = faiss.StandardGpuResources()
        except ImportError:
            raise("Faiss-gpu was not installed.")
        idx_gpu = faiss.GpuIndexFlatL2(self.resource, idx_cpu)
        self.idx_cpu = idx_cpu
        self.gpu_k = 2048
        super().__init__(idx_gpu, embedder)

    def add(self, X: Repertoire):
        hashes = self.embedder.transform(X).astype(np.float32)
        self.idx_cpu.add(hashes)
        self._add_hashes(hashes)
        self._add_ids(X)
        if isinstance(X, Repertoire):
            self._add_repertoire(X)
        return self
    def deallocate(self):
        """
        Deallocate index from GPU memory.
        """
        self.resource.noTempMemory()
        return self
    
    def _radius_search(self, x, d):
        """
        Radius search is not implemented for GPU indices. Hence, we perform a
        knn search with a fairly high k, and fall back on the cpu radius search
        if there are queries for which the furthest neighbor < radius.
        """
        radius = d**2 #faiss distances are squared
        k = min(self.idx.ntotal, self.gpu_k) # k for GPU knns, max supported is 2048

        # perform knn gpu search
        D, I = self.idx.search(x, k)

        # fall back on cpu radius search if not all neighbors<radius were in knn result
        mask = D[:, k - 1] < radius
        if mask.sum() > 0: # not all neighbors < radius in knn result
            lim_remain, D_remain, I_remain = self.idx_cpu.range_search(x[mask], radius)

        # merge results
        D_res, I_res = [], []
        nr = 0
        for i in range(len(x)):
            if not mask[i]:
                nv = (D[i, :] < radius).sum()
                D_res.append(D[i, :nv])
                I_res.append(I[i, :nv])
            else:
                l0, l1 = lim_remain[nr], lim_remain[nr + 1]
                D_res.append(D_remain[l0:l1])
                I_res.append(I_remain[l0:l1])
                nr += 1
        lims = np.cumsum([0] + [len(di) for di in D_res])
        D_res = np.hstack(D_res)
        I_res = np.hstack(I_res)
        return lims, np.sqrt(D_res), I_res 


class PynndescentIndex(BaseIndex):
    """
    Approximate search using PyNNDescent.
    """

    def __init__(
        self,
        embedder: Cdr3Embedder,
        k: int = 100,
        diversify_prob: float = 1.0,
        pruning_degree_multiplier: float = 1.5,
    ):
        """
        Initialize index.
        """
        from pynndescent import NNDescent

        idx = partial(
            NNDescent,
            n_neighbors=k,
            diversify_prob=diversify_prob,
            pruning_degree_multiplier=pruning_degree_multiplier,
        )
        idx.is_trained = True
        super().__init__(idx, embedder)


    def _add_hashes(self, X):
        self.idx = self.idx(X)
        self.idx.is_trained = True
    
    def _search(self, x, k):
        I, D = self.idx.query(x, k=k)
        return D,I

    def _search_self(self):
        I, D = self.idx.neighbor_graph
        return KnnResult(self.ids.values(), D, I, self.ids)


    def knn_search(self, y: Repertoire=None):
        """
        Search index for nearest neighbours.

        Parameters
        ----------
        y : Repertoire, optional
            The query TCRs. If not passed, returns the neighbours within the
            data added to the index, which is much faster.
        """
        if not y:
            return self._search_self()
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
        self, embedder: Cdr3Embedder, n_centroids: int = 32, n_probe: int = 5
    ) -> None:
        """
        Inverted file index for approximate nearest neighbour search.

        Parameters
        ----------
        embedder : Cdr3Embedder
            Fitted embedder object to transform CDR3 to vectors.
        n_centroids : int, default=32
            Number of centroids for the initial k-means clustering.
        n_probe : int, default=5
            Number of centroids to search at query time. Higher n_probe means
            higher recall, but slower speed.
        """
        idx = faiss.index_factory(embedder.m, f"IVF{n_centroids},Flat")
        super().__init__(idx, embedder)
        self.n_probe = n_probe


class HnswIndex(BaseIndex):
    def __init__(self, embedder: Cdr3Embedder, n_links: int = 32) -> None:
        """
        Index based on Hierarchical Navigable Small World networks.

        Parameters
        ----------
        embedder : Cdr3Embedder
            Fitted embedder object to transform CDR3 to vectors.
        n_links : int, default=32
            Number of bi-directional links created for each element during index
            construction. Increasing M leads to better recall but higher memory
            size and slightly slower searching.

        """
        idx = faiss.index_factory(embedder.m, f"HNSW{n_links},Flat")
        super().__init__(idx, embedder)


class FastApproximateIndex(BaseApproximateIndex):
    def __init__(
        self,
        embedder: Cdr3Embedder,
        n_centroids: int = 256,
        n_links: int = 32,
        n_probe: int = 10,
    ) -> None:
        """
        Approximate index based on a combination of IVF and HNSW using scalar
        quantizer encoding.

        Parameters
        ----------
        embedder : Cdr3Embedder
            Fitted embedder object to transform CDR3 to vectors.
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
        idx = faiss.index_factory(embedder.m, f"IVF{n_centroids}_HNSW{n_links},SQ6")
        super().__init__(idx, embedder)
        self.n_probe = n_probe

class RadiusSearchResult:
    """
    Result of radius search
    """
    def __init__(self, lims, D, I, query, ids, radius, rep=None) -> None:
        self.lims = lims
        self.D = D
        self.I = I
        self.ids = ids
        self.query = query
        self.rep = rep
        self.radius = radius

    def __repr__(self) -> str:
        return f"radius search result (query_size={len(self.query)}, radius={self.radius})"
    
    def to_df(self, add_result_info:bool=True) -> pd.DataFrame:

        query_cdr3s = self.query.cdr3s if isinstance(self.query, Repertoire) else list(self.query)

        res = pd.DataFrame({
            "query_sequence":query_cdr3s,
            "match":[self.ids[self.I[self.lims[i]:self.lims[i+1]]] for i in range(len(self.query))],
            "match_dist": [self.D[self.lims[i]:self.lims[i+1]] for i in range(len(self.query))],
        })

        if isinstance(self.query, Repertoire):
            res.insert(loc=0, column="query_sequence_id", value=self.query["sequence_id"].to_numpy())
        
        res = res.explode(["match", 'match_dist']).dropna().reset_index(drop=True)

        if add_result_info and self.rep is not None:
            index_indices = [self.I[self.lims[i]:self.lims[i+1]] for i in range(len(self.query))]
            index_indices = [a for b in index_indices for a in b] # indices of index matches
            res_right = self.rep.iloc[index_indices].reset_index(drop=True)
            res = pd.concat([res, res_right.add_prefix("match_")], axis="columns")

        return res
    

class KnnResult:
    """
    Result of k-nearest neighbor search.
    """

    def __init__(self, y, D, I, ids) -> None:
        self.D = np.sqrt(D)
        self.I = I
        self.query = y
        self.ids = ids
        self.query_size = D.shape[0]
        self.k = D.shape[1]

    def __repr__(self) -> str:
        return f"k-nearest neighbours result (size={self.query_size}, k={self.k})"
    
    @cached_property
    def _query_sequence_set(self) -> set:
        return set(self.ids)

    def _extract_neighbours(self, cdr3:str):
        try:
            i = self.y_idx[cdr3]
        except KeyError:
            raise KeyError(f"{cdr3} was not part of your query")
        I_ = np.vectorize(self._annotate_id)(self.I[i])
        return i, I_

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
        i, I_ = self._extract_neighbours(cdr3)
        return list(zip(I_, self.D[i]))

    def extract_neighbour_sequences(self, cdr3:str) -> List[str]:
        return self._extract_neighbours(cdr3)[1]

    def _annotate_id(self, cdr3_id):
        return self.ids.get(cdr3_id)

    def _refine_edges_iterator(self, distance_function, threshold):
        for s1 in self.y_idx.keys():
            seqs = self.extract_neighbour_sequences(s1)
            for match in ((s1, s2, dist) for s2 in seqs if (dist := distance_function(s1, s2)) <= threshold):
                yield match

    def _edges_iterator(self, threshold):
        pass


    def refine(self, distance_function:Callable, threshold:float, k:int=None) -> pd.DataFrame:
        """
        Perform a second round refinement of k-nearest neighbour matches, using a custom distance function and threshold.

        Parameters:
        distance_function : Callable,
            A function taking in two string arguments, returning the distance between the sequences.
        threshold : float
            Only sequence pairs at or below this distance are retained.
        k : int, optional
            Only the k closest matches for each query sequence are retained, if below the threshold.
        """
        df = pd.DataFrame(self._refine_edges_iterator(distance_function, threshold))
        if not df.empty:
            df.columns = ["query_cdr3", "match_cdr3", "distance"]
        if k:
            df = df.sort_values('distance').groupby("query_cdr3", sort=False).head(k).sort_index()
        return df

    def as_network(self, max_edge: float = 15):
        raise NotImplementedError()

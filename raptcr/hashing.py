from functools import lru_cache
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import MDS
from typing import Union

from .analysis import TcrCollection, Repertoire, Cluster
from .constants.base import AALPHABET
from .constants.hashing import BLOSUM_62, blosum_to_distance_matrix

from sklearn.utils.validation import check_is_fitted

def positional_encoding(sequence_len:int, m:int, p=10, max_rotations=0.5) -> np.ndarray:
    """
    Generate positional encoding based on sinus and cosinus functions with
    geometrically increasing wavelenghts.

    Parameters
    ----------
    sequence_len : int
        Number of AAs in the CDR3 sequence.
    m : int
        Number of output vector dimensions.
    p : int
        Exponent.
    max rotations : float
        Number of cycles the sine wave goes through.
    """
    distances = np.tile(np.arange(sequence_len)/sequence_len, (m,1)).T - np.tile(np.arange(m)/m, (sequence_len,1))
    cos_distances = np.cos(distances*2*np.pi*max_rotations)
    pow_cos_distances = np.power(cos_distances, p)
    return pow_cos_distances

class Cdr3Hasher(BaseEstimator, TransformerMixin):
    def __init__(self, distance_matrix:np.array=blosum_to_distance_matrix(BLOSUM_62), m:int=16, p:float=10.0, max_rotations:float=2.0, position_weight:float=1.0, trim_left:int=0, trim_right:int=0) -> None:
        """
        Locality-sensitive hashing for amino acid sequences. Hashes CDR3
        sequences of varying lengths into m-dimensional vectors, preserving
        similarity.

        Parameters
        ----------
        """
        self.m = m
        self.distance_matrix = distance_matrix
        self.position_weight = position_weight
        self.max_rotations = max_rotations
        self.p = p
        self.trim_left = trim_left
        self.trim_right = trim_right

    def fit(self, X=None, y=None):
        """
        Fit Cdr3Hasher. This method generates the necessary components for hashing.
        """
        self.aa_vectors_ = self._construct_aa_vectors()
        self.position_vectors_ = self._construct_position_vectors()
        return self

    def _construct_aa_vectors(self) -> dict:
        """
        Create dictionary containing AA's and their corresponding hashes, of
        type str : np.ndarray[m].
        """
        vecs = MDS(n_components=self.m, dissimilarity="precomputed").fit_transform(self.distance_matrix)
        vecs_dict = {aa:vec for aa,vec in zip(AALPHABET, vecs)}
        return vecs_dict

    def _construct_position_vectors(self) -> dict:
        """
        Create positional encoding matrix for different CDR3 lengths

        Returns
        -------
        dict int : np.ndarray[m, l]
            Dict of encoding matrices for each CDR3 sequence length l.
        """
        position_vectors = dict()
        for cdr3_length in range(1,50): # maximum hashable sequence length = 50
            vector = positional_encoding(cdr3_length, self.m, self.p, self.max_rotations)
            scaled_vector = 1 + self.position_weight*(vector-1)
            position_vectors[cdr3_length] = scaled_vector
        return position_vectors

    def _pool(self, aa_vectors:np.ndarray, position_vectors:np.ndarray) -> np.ndarray:
        """
        Pool position-encoded AA hashes along y-axis.
        """
        return np.multiply(aa_vectors, position_vectors).sum(axis=0)


    @lru_cache(maxsize=None)
    def _hash_cdr3(self, cdr3: str) -> np.array:
        """
        Generate hash from AA sequence. Results are cached.

        Parameters
        ----------
        cdr3 : str
            The amino acid CDR3 sequence to hash.

        Returns
        -------
        np.array[m,]
            The resulting hash vector.
        """
        # trim CDR3 sequence
        r = -self.trim_right if self.trim_right else len(cdr3)
        cdr3 = cdr3[self.trim_left:r]
        # hash
        l = len(cdr3)
        aas = np.array([self.aa_vectors_[aa] for aa in cdr3])
        pos = self.position_vectors_[l]
        return self._pool(aas, pos)

    def _hash_collection(self, seqs: TcrCollection) -> np.array:
        """
        Generate hash for a group of CDR3 sequences, e.g. a TCR cluster or TCR repertoire.

        Parameters
        ----------
        cluster : TcrCollection, Iterable
            Cluster object containing sequences to hash.

        Returns
        -------
        np.array[64,]
            The resulting hash vector.
        """
        sequence_hashes = [self._hash_cdr3(cdr3) for cdr3 in seqs.cdr3s]
        return np.mean(sequence_hashes, axis=0)

    def transform(self, X: Union[TcrCollection, list, str], y=None) -> np.array:
        """
        Generate CDR3 hashes.

        Parameters
        ----------
        x : Union[TcrCollection, list, str]
            Objects to hash, this can be a single CDR3 sequence, but also a
            TcrCollection subclass or list thereof.

        Returns
        -------
        np.array[n,m]
            Array containing m-dimensional hashes for each of the n provided inputs.
        """
        check_is_fitted(self)
        match X:
            case Repertoire() | list() | np.ndarray():
                return np.array([self.transform(s) for s in X]).astype(np.float32)
            case Cluster():
                return self._hash_collection(X)
            case _:
                return self._hash_cdr3(X)
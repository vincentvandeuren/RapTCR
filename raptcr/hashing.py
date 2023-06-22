from functools import lru_cache
from typing import Union, Tuple
from abc import ABC

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import MDS
from sklearn.utils.validation import check_is_fitted

from .analysis import Repertoire
from .constants.base import AALPHABET
from .constants.hashing import DEFAULT_DM


def positional_encoding(sequence_len: int, m: int, n_rotations=4.5, p=0.039) -> np.ndarray:
    """
    Generate positional encoding cosinus functions with increasing wavelenghts.

    Parameters
    ----------
    sequence_len : int
        Number of AAs in the CDR3 sequence.
    m : int
        Number of output vector dimensions. Must be a perfect square number.
    p : float
        Nonzero, positive number used as the base for the geometric series.
    n_rotations : float
        Maximal frequency of the cosine functions.
    """
    d = np.outer(
        np.power(p, np.arange(m) / m),  # geometrically decreasing
        np.arange(1, sequence_len + 1)
        / sequence_len,  # linear ]0, 1] interval with steps 1/sequence_len
    )
    cos_d = np.cos(d * np.pi * 2 * n_rotations)
    return cos_d


class BaseCdr3Embedder(ABC):
    def __init__(
            self,
            aa_dims:int,
            pos_dims:int,
            distance_matrix: np.ndarray,
            n_rotations: float,
            p: float,
            trim_left: int,
            trim_right: int,
            ) -> None:
        self.aa_dims = aa_dims
        self.pos_dims = pos_dims
        self.trim_left = trim_left
        self.trim_right = trim_right
        self.distance_matrix = distance_matrix
        self.n_rotations = n_rotations
        self.p = p
        super().__init__()

    def _construct_aa_vectors(self) -> dict:
        """
        Create dictionary containing AA's and their corresponding hashes, of
        type str : np.ndarray[m].
        """
        vecs = MDS(
            n_components=self.aa_dims,
            dissimilarity="precomputed",
            random_state=11,
            normalized_stress=False,
            eps=1e-6,
        ).fit_transform(self.distance_matrix)
        vecs_dict = dict(zip(AALPHABET, vecs))
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
        for cdr3_length in range(1, 50):  # maximum hashable sequence length = 50
            vector = positional_encoding(cdr3_length, self.pos_dims, self.n_rotations, self.p)
            position_vectors[cdr3_length] = vector.T
        return position_vectors
    
    def _trim_sequence(self, sequence:str) -> str:
        r = -self.trim_right if self.trim_right else len(sequence)
        return sequence[self.trim_left : r]
    
    def _embed_sequence(self, sequence:str) -> np.ndarray:
        """
        Generate vector from AA sequence.

        Parameters
        ----------
        cdr3 : str
            The amino acid CDR3 sequence to hash.

        Returns
        -------
        np.array[m,]
            The resulting hash vector.
        """
        sequence_trimmed = self._trim_sequence(sequence)
        aas = np.array([self.aa_vectors_[aa] for aa in sequence_trimmed])
        pos = self.position_vectors_[len(sequence_trimmed)]
        return (pos.T @ aas).flatten()


class Cdr3Embedder(BaseEstimator, TransformerMixin, BaseCdr3Embedder):
    def __init__(
            self,
            aa_dims:int=24,
            pos_dims:int=10,
            distance_matrix: np.ndarray = DEFAULT_DM,
            n_rotations: float = 4.5,
            p: float = 0.039,
            trim_left: int = 0,
            trim_right: int = 0
    ) -> None:
        """
        Simple CDR3 embedding for amino acid sequences. Hashes CDR3 sequences of
        varying lengths into m-dimensional vectors, preserving similarity.

        Parameters
        ----------
        distance_matrix : np.ndarray[20,20]
            Twenty by twenty matrix containing AA distances. Amino acids are
            ordered alphabetically  (see AALPHABET constant)
        m : int
            Number of embedding dimensions.
        p : int
            Positional importance scaling factor.
        trim_left : int
            Number of amino acids trimmed from left side of sequence.
        trim_right : int
            Number of amino acids trimmed from right side of sequence.
        """
        super().__init__(aa_dims, pos_dims, distance_matrix, n_rotations, p, trim_left, trim_right)

    def __repr__(self):
        return f"Cdr3Embedder(m={self.m})"

    def fit(self, X=None, y=None):
        """
        Fit Cdr3Hasher. This method generates the necessary components for hashing.
        """
        self.aa_vectors_ = self._construct_aa_vectors()
        self.position_vectors_ = self._construct_position_vectors()
        return self

    @property
    def m(self):
        return (self.aa_dims * self.pos_dims)

    def transform(self, X: Union[Repertoire, list, str], y=None) -> np.array:
        """
        Generate CDR3 embeddings.

        Parameters
        ----------
        x : Union[TcrCollection, list, str]
            Objects to embed, this can be a single CDR3 sequence, but also a
            TcrCollection subclass or list thereof.

        Returns
        -------
        np.array[n,m]
            Array containing m-dimensional hashes for each of the n provided inputs.
        """
        check_is_fitted(self)
        if isinstance(X, str):
            return self._embed_sequence(self._trim_sequence(X))
        elif isinstance(X, (list, np.ndarray)):
            return np.array([self.transform(s) for s in X]).astype(np.float32)
        elif isinstance(X, (Repertoire, pd.DataFrame)):
            return np.array([self.transform(s) for s in X.cdr3s]).astype(np.float32)
        # elif isinstance(X, Cluster):
        #     return self._hash_collection(X)
        # elif isinstance(X, ClusteredRepertoire):
        #     return np.vstack([self.transform(s) for s in X]).astype(np.float32)
        else:
            raise ValueError("Incorrect type")
        
class PairedChainCdr3Embedder(BaseEstimator, TransformerMixin):
    def __init__(self, alpha_embedder:Cdr3Embedder, beta_embedder:Cdr3Embedder) -> None:
        """
        Embedder for paired alpha-beta TCR sequences.

        Parameters
        ----------
        alpha_embedder : Cdr3Embedder
            The initialized Cdr3 Embedder class used for the alpha chain.
        beta_embedder : Cdr3Embedder
            The initialized Cdr3 Embedder class used for the beta chain.
        
        Notes
        -----
        Use different embedding dimensions for alpha and beta to weight the
        chains accordingly. By default, the alpha and beta chain are encoded in
        80 and 240 dimensions, respectively; resulting in a 1:3 alpha:beta
        weight ratio.
        """
        self.alpha_embedder = alpha_embedder
        self.beta_embedder = beta_embedder
        super().__init__()

    def __repr__(self) -> str:
        m_a = self.alpha_embedder.aa_dims * self.alpha_embedder.pos_dims
        m_b = self.beta_embedder.aa_dims * self.beta_embedder.pos_dims______
        return f"PairedChainCdr3Embedder(m_alpha={m_a}, m_beta={m_b})"
    

    def fit(self, X=None, y=None):
        """
        Fit PairedChainCdr3Embedder. This method generates the necessary components for hashing.
        """
        self.alpha_embedder.fit(X, y)
        self.beta_embedder.fit(X, y)
        self.is_fitted_ = True
        return self
    
    def _embed_sequence(self, alpha_sequence:str, beta_sequence:str) -> np.ndarray:
        """
        Generate vector from AA sequence.

        Parameters
        ----------
        cdr3 : str
            The amino acid CDR3 sequence to hash.

        Returns
        -------
        np.array[m,]
            The resulting hash vector.
        """
        return np.concatenate([
            self.alpha_embedder.transform(alpha_sequence),
            self.beta_embedder.transform(beta_sequence)
        ])
    
    def transform(self, X: Union[Repertoire, np.ndarray, Tuple[str,str]], y=None):
        if isinstance(X, Tuple):
            return self._embed_sequence(*X)
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, np.ndarray):
            if X.shape[1] == 2:
                return np.array([self.transform(tuple(t)) for t in X])


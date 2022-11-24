from bitarray import bitarray
from bitarray.util import urandom, zeros
from functools import lru_cache
import numpy as np
import pandas as pd
import pandas as pd
from typing import Union

from .analysis import TcrCollection
from .constants.hashing import AA_HASHES

class Cdr3Hasher():

    def __init__(self, pos_p:float=0.5, clip:int=0, aa_base_hashes:dict=None) -> None:
        """
        Locality-sensitive hashing for amino acid sequences. Hashes CDR3
        sequences of varying lengths into 64-dimensional vectors, preserving
        similarity.

        Parameters
        ----------
        pos_p : float
            The relative importance of AA position for constructing the hash. A
            value 0.4-0.6 showed best performance.
        clip : int, default = 0
            Clip the CDR3s with size n, thus ignoring first n and last n AAs.
        aa_base_hashes : dict, optional
            Optionally provide custom base hashes for each amino acid.
        
        """
        if not aa_base_hashes:
            aa_base_hashes = AA_HASHES
        self.aa_hashes = aa_base_hashes
        self.pos_hashes = self._construct_pos_hashes(pos_p)
        self.clip = clip

    def _construct_pos_hashes(self, pos_p) -> None:
        """
        Construct the position hash bitstring building blocks.

        pos_p : float
            The relative importance of AA position for constructing the hash. In
            practice, pos_p represents the probability of a random bit flip
            happening between each of the 360 subsequent bitarrays.

        TODO: Hardcode these once optimized, remove randomness.
        """
        def _mutate_random(s):
            i = np.random.randint(0,64)
            s[i] = not s[i]
            return s

        position_arr = {}
        s = urandom(64)
        for n in range(0,360):
            for _ in range(int(pos_p)):
                s = _mutate_random(s)
            if np.random.random() <= pos_p-int(pos_p):
                s = _mutate_random(s)
            position_arr[n] = s.copy()
        return position_arr

    @lru_cache(maxsize=None)
    def _hash_cdr3(self, cdr3:str) -> np.array:
        """
        Generate hash from AA sequence. Results are cached.

        Parameters
        ----------
        cdr3 : str
            The amino acid CDR3 sequence to hash.
        
        Returns 
        -------
        np.array[64,]
            The resulting hash vector.
        """
        if self.clip:
            cdr3 = cdr3[self.clip:-self.clip]
        cdr3_len = len(cdr3)
        hashes = [self.aa_hashes[aa]^self.pos_hashes[int(359*i/(cdr3_len-1))] for i,aa in enumerate(cdr3)]
        return self._sum_hashlist(hashes)

    def _hash_collection(self, seqs:TcrCollection) -> np.array:
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
        ch = [self.hash_cdr3(cdr3) for cdr3 in seqs.cdr3s]
        return np.mean(ch, axis=0)

    def hash(self, x:Union[TcrCollection, list, str]) -> np.array:
        """
        Generate CDR3 hashes.

        Parameters
        ----------
        x : Union[TcrCollection, list, str]
            Objects to hash, this can be a single CDR3 sequence, but also a
            TcrCollection subclass or list thereof.
        
        Returns
        -------
        np.array[n,64]
            Array containing 64-dimensional hashes for each of the n provided inputs.
        """

        match x:
            case TcrCollection(): return self._hash_collection(x)
            case list(): return np.array(self.hash(y) for y in x)
            case _: return self._hash_cdr3(x)

    def _sum_hashlist(self, hashlist) -> np.array:
        mh = np.array([h.tolist() for h in hashlist])
        mh[mh==0] = -1
        mh = mh.sum(axis=0)
        return mh

    def hash_cdr3_kmerized(self, cdr3:str, k:int):
        """Experimental! Use subsequences of length k (k-mers) as the AA-hash basis. 
        """
        def _kmer_iterator(s, k):
            for i in range(len(s)-k+1):
                yield (i, s[i:i+k])

        hashlist = []
        for i,kmer in _kmer_iterator(cdr3, k):
            zxor = zeros(64)
            [zxor := zxor ^ self.aa_hashes[aa]<<i for i,aa in enumerate(kmer)]
            hashlist.append(zxor.copy() ^ self.pos_hashes[i])
        return self._sum_hashlist(hashlist)
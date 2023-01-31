import time
from .constants.base import AALPHABET

from collections import Counter
import numpy as np

def timed(myfunc):
    # Decorator to keep track of time required to run a function
    def timed(*args, **kwargs):
        start = time.time()
        result = myfunc(*args, **kwargs)
        end = time.time()
        print(f"Total time to run '{myfunc.__name__}': {(end-start):.3f}s")
        return result

    return timed

def profile_matrix(sequences : list):
    '''
    Calculates the profile matrix for a set of sequences (i.e. all cluster members).
    NOTE: this version does not take into account the expected frequency of each amino acid at each position.
    '''

    # Make sure to proceed only if all sequences in the cluster have equal length
    seq_len = len(sequences[0])
    if not all(len(seq) == seq_len for seq in sequences):

        # On the rare occasion that a cluster contains sequences of inequal length.
        # Typically, there is/are only one (or very few) sequence(s) that differ from the avg. CDR3 length in the cluster.
        # Therefore, we use the length of the highest proportion of sequences as the standard, and delete all others.
        seq_len = Counter([len(s) for s in sequences]).most_common()[0][0]
        sequences = [s for s in sequences if len(s) == seq_len]
    
    # Initiate profile matrix with zeros
    pm = np.zeros(shape=(len(AALPHABET), seq_len))

    # initiate AA dict:
    AAs = {aa: i for i, aa in enumerate(AALPHABET)}

    # Fill in profile matrix with counts
    for s in sequences:
        for i, aa in enumerate(s):
            pm[AAs[aa], i] += 1

    # normalize profile matrix to percentages
    pm = pm / len(sequences)

    return pm


def motif_from_profile(profile, method, cutoff=.7):
    '''
    Generate consensus sequence motif from a profile matrix.
    Square brackets [...] indicate multiple aa possibilities at that position.
    X represents any aa.
    '''
    AA_map = {i:aa for i,aa in enumerate(AALPHABET)}

    consensus = ''
    
    if method.lower() == 'standard':
        top_idxs = np.argpartition(profile, -2, axis=0)[-2:].T
        top_values = np.partition(profile, -2, axis=0)[-2:].T
        for (second_max_idx, max_idx), (second_max_value, max_value) in zip(top_idxs, top_values):
            if max_value >= cutoff:
                consensus += AA_map[max_idx]
            elif max_value + second_max_value >= cutoff:
                if max_value >= 2*second_max_value:
                    consensus += AA_map[max_idx].lower()
                else:
                    consensus += f"[{AA_map[max_idx]}{AA_map[second_max_idx]}]"
            else:
                consensus += "."
                
    elif method.lower() == 'conservative':
        max_idx, max_value = np.argmax(profile.T, axis=1), np.amax(profile.T, axis=1)
        for idx, value in zip(max_idx, max_value):
            if value > cutoff:
                consensus += AA_map[idx]
            else:
                consensus += "."

    return consensus
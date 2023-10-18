import pandas as pd

from .analysis import Repertoire
from .constants.base import AALPHABET

def _is_productive(seq:str) -> bool:
    return all([aa in AALPHABET for aa in seq])


def read_AIRR(
    filepath: str,
    filter_productive: bool = True,
    filter_TRB: bool = False,
    filter_min_duplicate_count: int = 0,
) -> Repertoire:
    """
    Read in data from an AIRR formatted `.tsv` file.

    Parameters
    ----------
    filepath : str
        Path of file.
    filter_productive : bool, default = True
        Retain only productive sequences.
    filter_TRB : bool, default = True
        Retain only TCR beta chains, inferred from the V and J-gene calls.
    filter_min_duplicate_count : int, optional
        Retain only sequences observed more than n times.


    Returns
    -------
    Repertoire
        Repertoire object containing the data.
    """

    cols = [
        "sequence_id",
        "productive",
        "v_call",
        "j_call",
        "duplicate_count",
        "junction_aa",
    ]

    available_cols = pd.read_csv(filepath, sep="\t", nrows=0).columns.to_list()
    if all([col in available_cols for col in cols]):
        df = pd.read_csv(filepath, sep="\t", usecols=cols)
    else:
        df = pd.read_csv(filepath, sep="\t", usecols=available_cols)

    if "sequence_id" in df:
        df = df.set_index("sequence_id")

    if "productive" in df:
        if df["productive"].dtype == "O":
            df["productive"] = df["productive"] == "T"

        if filter_productive:
            df = df.query("productive == True")

    if filter_min_duplicate_count:
        df = df.query(f"duplicate_count > {filter_min_duplicate_count}")

    if filter_TRB:
        df = df.query('v_call.str.contains("TRB") or j_call.str.contains("TRB")')

    return Repertoire(df)


def read_OLGA(filepath: str) -> Repertoire:
    """
    Read in data from an OLGA-generated `.tsv` file.

    Parameters
    ----------
    filepath : str
        Path of file.

    Returns
    -------
    Repertoire
        Repertoire object containing the data.
    """
    df = pd.read_csv(filepath, sep="\t", header=None)
    df = df.drop(0, axis=1)
    df.columns = ["junction_aa", "v_call", "j_call"]
    return Repertoire(df)

def read_vdjdb(filepath:str, filter_TRB:bool = False, filter_human : bool = True, exclude_10x:bool=False, exclude_studies:list=[], min_score:int=None) -> Repertoire:
    """
    Read the vdjdb.slim.txt file into a Repertoire object

    Parameters
    ----------
    filepath : str
        Location of the vdjdb `vdjdb.slim.txt` database file. 
    filter_TRB : bool, default = True
        Retain only TRB sequences.
    filter_human : bool, default = True
        Retain only human-derived sequences.
    exclude_10x : bool, optional
        Exclude sequence annotations derived only from the 10X genomics "A new
        way of exploring immunity" study.
    exclude_studies : list, optional
        List containing reference_ids of other studies to exclude.
    min_score : int, optional
        Minimum VDJDB score to be included.
    """
    df =  pd.read_csv(filepath, sep="\t")
    df.columns = df.columns.str.replace(".", "_", regex=False)

    if exclude_10x:
        exclude_studies.append('https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#')

    query_filter = '(reference_id not in @exclude_studies)'
    if filter_TRB:
        query_filter += ' and (gene == "TRB")'
    if filter_human:
        query_filter += ' and (species == "HomoSapiens")'
    if min_score:
        query_filter += f' and (vdjdb_score >= {min_score})'

    df = (
        df
        .query(query_filter)
        .rename({"v_segm":"v_call", "j_segm":"j_call", "cdr3":"junction_aa"}, axis="columns")
        )

    return Repertoire(df)


def read_mixcr(filepath:str, filter_TRB:bool=False, filter_productive:bool=False, filter_min_duplicate_count:bool=0, read_extra_cols:list=[]):
    """
    Read in data from an MiXCR formatted `.txt` file.

    Parameters
    ----------
    filepath : str
        Path of file.
    read_extra_cols:  list, optional
        List of columns to read in, in addition to the default (required) ones.
    filter_productive : bool, default = True
        Retain only productive sequences.
    filter_TRB : bool, default = False
        Retain only TCR beta chains, inferred from the V and J-gene calls.
    filter_min_duplicate_count : int, optional
        Retain only sequences observed more than n times.


    Returns
    -------
    Repertoire
        Repertoire object containing the data.
    """

    column_map = {
        "cloneId":"sequence_id",
        "cloneCount":"duplicate_count",
        "nSeqCDR3":"junction",
        "aaSeqCDR3":"junction_aa",
        "allVHitsWithScore":"v_call",
        "allJHitsWithScore":"j_call",
    }

    column_map.update(dict(zip(read_extra_cols, read_extra_cols)))
    
    df = pd.read_csv(filepath, sep="\t", usecols=column_map.keys())
    df = df.rename(column_map, axis="columns")

    df["v_call"] = df["v_call"].str.split("(").str[0]
    df["j_call"] = df["j_call"].str.split("(").str[0]

    df["v_call"] = df["v_call"].str.replace("DV", "/DV").str.replace("//", "/")
    
    df["productive"] = df["junction_aa"].apply(_is_productive)

    if filter_productive:
            df = df.query("productive == True")

    if filter_min_duplicate_count:
        df = df.query(f"duplicate_count > {filter_min_duplicate_count}")

    if filter_TRB:
        df = df.query('v_call.str.contains("TRB") or j_call.str.contains("TRB")')

    return Repertoire(df)


def read_clustcr(filepath: str) -> Repertoire:
    raise NotImplementedError()

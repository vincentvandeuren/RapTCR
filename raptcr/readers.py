import pandas as pd

from .analysis import Repertoire


def read_AIRR(
    filepath: str,
    filter_productive: bool = True,
    filter_TRB: bool = True,
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
    df = pd.read_csv(filepath, sep="\t", usecols=cols)
    df = df.set_index("sequence_id")

    if "productive" in df:
        if df["productive"].dtype == "O":
            df["productive"] = df["productive"] == "T"

    if filter_productive:
        df = df.query("productive == True")

    if filter_min_duplicate_count:
        df = df.query(f"duplicate_count > {filter_min_duplicate_count}")

    df = df.drop("productive", axis=1)

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

def read_vdjdb(filepath:str, filter_TRB:bool = True, filter_human : bool = True, exclude_10x:bool=False, exclude_studies:list=[]) -> Repertoire:
    """
    Read the vdjdb.slim.txt file into a Repertoire object

    Parameters
    ----------
    filepath : str
        Location of the vdjdb database file.
    filter_TRB : bool, default = True
        Retain only TRB sequences.
    filter_human : bool, default = True
        Retain only human-derived sequences.
    exclude_10x : bool, optional
        Exclude sequence annotations derived only from the 10X genomics "A new way of exploring immunity" study.
    exclude_studies : list, optional
        List containing reference_ids of other studies to exclude.
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

    df = (
        df
        .query(query_filter)
        .rename({"v_segm":"v_call", "j_segm":"j_call", "cdr3":"junction_aa"}, axis="columns")
        )

    return Repertoire(df)


def read_clustcr(filepath: str) -> Repertoire:
    raise NotImplementedError()

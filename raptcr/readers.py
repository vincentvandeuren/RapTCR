import pandas as pd

from analysis import Repertoire


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

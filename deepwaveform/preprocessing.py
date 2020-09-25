import pandas as pd
import numpy as np


def load_dataset(filepath, wv_cols=list(range(64)), inv_z=True):
    """Loads a dataset of waveforms into a pandas dataframe. Required
    columns names are id,class,x,y,z,0,...,199, where "id" marks the index,
    "class" the classification (int), "x", "y", "z" the position of each point
    and 0,...,199 the corresponding waveform. The file format is given as a
    csv with seperator " ".

    :param filepath: Path to file
    :type filepath: str
    :param wv_cols: Column names of waveforms, defaults to list(range(64))
    :type wv_cols: List, optional
    :param inv_z: Whether the z-axis should be inverted, defaults to True
    :type inv_z: bool, optional
    :return: a pandas dataframe
    :rtype: pandas.DataFrame
    """
    names = ["id", "class", "x", "y", "z"] + list(range(200))
    data = pd.read_csv(filepath, sep=" ", header=None, names=names)
    data = data.drop(set(range(200)).difference(wv_cols), axis=1)
    # we need to invert z
    if inv_z:
        data["z"] = max(data["z"])-data["z"]
    return data


def waveform2matrix(df, wv_cols=list(range(64))):
    """Takes a dataframe containing waveforms and returns a numpy
    matrix containing the waveforms.

    :param df: Dataframe containing waveforms
    :type df: pandas.DataFrame
    :param wv_cols: Column names of waveforms, defaults to list(range(64))
    :type wv_cols: List, optional
    :return: a [#samples]x[#waveform_dimension]-matrix
    :rtype: np.ndarray
    """
    # this matrix will make it easier to plot samples
    mat = np.zeros((df.shape[0], len(wv_cols)))
    for idx, col in enumerate(wv_cols):
        mat[:, idx] = df[col].values
    return mat

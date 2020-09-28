import pandas as pd
import numpy as np

def load_dataset(filepath):
    """Loads a dataset of waveforms into a pandas dataframe.".

    :param filepath: Path to file
    :type filepath: str
    :param inv_z: Whether the z-axis should be inverted, defaults to True
    :type inv_z: bool, optional
    :return: a pandas dataframe
    :rtype: pandas.DataFrame
    """
    data = pd.read_csv(filepath, sep=";", index_col="index")
    return data


def waveform2matrix(df, wv_cols=list(map(str, range(64)))):
    """Takes a dataframe containing waveforms and returns a numpy
    matrix containing the waveforms.

    :param df: Dataframe containing waveforms
    :type df: pandas.DataFrame
    :param wv_cols: Column names of waveforms, defaults to list(map(str, range(64)))
    :type wv_cols: List, optional
    :return: a [#samples]x[#waveform_dimension]-matrix
    :rtype: np.ndarray
    """
    # this matrix will make it easier to plot samples
    mat = np.zeros((df.shape[0], len(wv_cols)))
    for idx, col in enumerate(wv_cols):
        mat[:, idx] = df[col].values
    return mat

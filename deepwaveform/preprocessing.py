import pandas as pd
import numpy as np


def load_dataset(filepath, samplesize=None, chunksize=None):
    """Returns an iterator over chunks of a dataset of waveforms as pandas dataframes. Separator is ';'. Dataframe must be indexed by 'index'-column".

    :param filepath: Path to file
    :type filepath: str
    :param samplesize: Sample size. if None, then all is loaded
    :type samplesize: int
    :param chunksize: Size of loaded chunks. if None, then all is loaded in one chunk
    :type chunksize: int
    :return: a pandas dataframe
    :rtype: pandas.DataFrame
    """
    skip = None
    if samplesize is not None:
        import random
        row_count = sum(1 for line in open(filepath))-1
        skip = sorted(random.sample(range(1,row_count+1),row_count-samplesize))

    call = pd.read_csv(filepath,
                    sep=";",
                    index_col="index",
                    skiprows=skip,
                    chunksize=chunksize)

    if chunksize is not None:
        for chunk in call:
            yield chunk
    else:
        yield call


def waveform2matrix(df, wv_cols=list(map(str, range(64)))):
    """Takes a dataframe containing waveforms and returns a numpy
    matrix containing the waveforms.

    :param df: Dataframe containing waveforms
    :type df: pandas.DataFrame
    :param wv_cols: Column names of waveforms,
        defaults to list(map(str, range(64)))
    :type wv_cols: List, optional
    :return: a [#samples]x[#waveform_dimension]-matrix
    :rtype: np.ndarray
    """
    # this matrix will make it easier to plot samples
    mat = np.zeros((df.shape[0], len(wv_cols)))
    for idx, col in enumerate(wv_cols):
        mat[:, idx] = df[col].values
    return mat

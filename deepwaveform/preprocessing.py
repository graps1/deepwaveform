import pandas as pd
import numpy as np


def load_dataset(filepath, wv_cols=list(range(64)), inv_z=True):
    names = ["id", "class", "x", "y", "z"] + list(range(200))
    data = pd.read_csv(filepath, sep=" ", header=None, names=names)
    data = data.drop(set(range(200)).difference(wv_cols), axis=1)
    # we need to invert z
    if inv_z:
        data["z"] = max(data["z"])-data["z"]
    return data


def waveform2matrix(df, wv_cols=list(range(64))):
    # this matrix will make it easier to plot samples
    mat = np.zeros((df.shape[0], len(wv_cols)))
    for idx, col in enumerate(wv_cols):
        mat[:, idx] = df[col].values
    return mat

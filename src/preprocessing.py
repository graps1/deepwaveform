import pandas as pd
import numpy as np


def load_dataset(filepath, wv_dim=200):
    names = ["id", "class", "x", "y", "z"] + [i for i in range(0, wv_dim)]
    data = pd.read_csv(filepath, sep=" ", header=None, names=names)
    # we need to invert z
    data["z"] = max(data["z"])-data["z"]
    return data


def waveform2matrix(df, wv_dim=200):
    # this matrix will make it easier to plot samples
    mat = np.zeros((df.shape[0], wv_dim))
    for i in range(wv_dim):
        mat[:, i] = df[i].values
    return mat

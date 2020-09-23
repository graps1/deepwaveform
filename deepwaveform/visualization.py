import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_rgba

from deepwaveform import waveform2matrix


def plot_waveforms(df,
                   classcol="class",
                   class_label_mapping=["Land", "Water"],
                   wv_cols=list(range(64))):

    # get waveforms in matrix form
    waveforms = waveform2matrix(df, wv_cols=wv_cols)
    # set up figure
    fig = plt.figure(figsize=(23, 7))
    ax = fig.add_subplot(111)
    ax.set_title("Waveforms")
    ax.set_xlabel("Index")
    ax.set_ylabel("Amplitude")
    # x-values
    xs = np.arange(len(wv_cols))
    # plot each sampled waveform
    occurred = set({})
    for idx in range(waveforms.shape[0]):
        # find the index in the waveform where it drops to 0
        first_0_idx = np.where(waveforms[idx, :] == 0)[0]
        first_0_idx = first_0_idx[0] if len(first_0_idx) != 0 else\
            len(wv_cols)-1
        # the classifiaction (land or water) of this point
        dataclass = df[classcol][idx]
        # check if we already have a label for this class
        label = None
        if dataclass not in occurred:
            label = class_label_mapping[dataclass]
            occurred.add(dataclass)
        colors = ["g", "b", "r", "y", "b"]
        # plot full (colored) waveform (stripped of 0-elements)
        plt.plot(xs[:first_0_idx],
                 waveforms[idx, :first_0_idx],
                 c=colors[dataclass],
                 label=label)
    plt.legend(loc='upper left')
    plt.show()


def plot_pcl(df,
             plotsize=20,
             targetcol="class",
             colormap=cm.coolwarm,
             xcol="x",
             ycol="y",
             zcol="z"):
    # initialize 3d-plot
    fig = plt.figure(figsize=(plotsize, plotsize))
    ax = fig.add_subplot(111, projection='3d')
    norm = Normalize(min(df[targetcol]), max(df[targetcol]))

    scatter = ax.scatter(df[xcol], df[ycol], df[zcol],
                         c=df[targetcol], cmap=colormap, norm=norm,
                         marker=".", s=0.7)
    legend = ax.legend(*scatter.legend_elements(),
                       loc="lower left", title="Classes")
    ax.add_artist(legend)

    ax.grid(False)
    ax.set_axis_off()
    ax.set_xlim3d(-60, 60)
    ax.set_ylim3d(-60, 60)
    ax.set_zlim3d(-60, 60)
    ax.view_init(elev=40., azim=-120.)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_pcl_prediction(df,
                        plotsize=20,
                        probabilities_col=["Land", "Water"],
                        colors=["green", "blue"],
                        xcol="x",
                        ycol="y",
                        zcol="z",
                        sigmoid_scaling=False,
                        sigmoid_scaling_factor=1):
    def sigmoid(x):
        return 1/(np.exp(-x)+1)
    probarr = np.array(df[probabilities_col])
    if sigmoid_scaling:
        probarr = sigmoid(sigmoid_scaling_factor*(2*probarr-1))

    # linear interpolation between colors
    rgba_colors = np.array([to_rgba(color) for color in colors])
    colorsarr = np.zeros(shape=(df.shape[0], 4))
    for idx, col in enumerate(rgba_colors):
        colorsarr += (probarr[:, idx]*np.array([col]*df.shape[0]).T).T
    colorsarr = (colorsarr - np.min(colorsarr))/(
        np.max(colorsarr) - np.min(colorsarr))

    # initialize 3d-plot
    fig = plt.figure(figsize=(plotsize, plotsize))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[xcol], df[ycol], df[zcol],
               color=colorsarr,
               marker=".", s=0.7)

    for color, classname in zip(colors, probabilities_col):
        ax.add_line(plt.Line2D([0], [0], color=color, label=classname, lw=4))

    legend = ax.legend(loc="lower left", title="Classes")
    ax.add_artist(legend)

    ax.grid(False)
    ax.set_axis_off()
    ax.set_xlim3d(-60, 60)
    ax.set_ylim3d(-60, 60)
    ax.set_zlim3d(-60, 60)
    ax.view_init(elev=40., azim=-120.)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_training_progress(stats, figsize=(12, 6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    xs = np.arange(len(stats))
    ys = np.array([stat["meanloss"] for stat in stats])
    ys_sd = np.array([stat["varloss"] for stat in stats])**0.5
    ax.plot(xs, ys, color="red")
    ax.fill_between(xs, ys-0.5*ys_sd, ys+0.5*ys_sd, color="red", alpha=0.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean loss")
    plt.show()


def plot_confusion_matrix(model,
                          dataset,
                          class_label_mapping=["Land", "Water"]):
    wf, lab = dataset[:]["waveform"], dataset[:]["label"].numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cf_mat = np.zeros(shape=(model.output_dimension, model.output_dimension))

    for idx, row in enumerate(model.predict(wf)):
        predicted = np.argmax(row, axis=0)
        correct = lab[idx]
        cf_mat[predicted, correct] += 1

    for i in range(model.output_dimension):
        cf_mat[:, i] = cf_mat[:, i]/np.sum(cf_mat[:, i])

    for i in range(model.output_dimension):
        for j in range(model.output_dimension):
            p = cf_mat[i, j]
            ax.text(j, i, "%.3f" % p, va="center", ha="center")

    ax.matshow(cf_mat, cmap=cm.coolwarm)
    ax.set_yticks(range(cf_mat.shape[0]))
    ax.set_yticklabels(["%s (predicted)" % class_label_mapping[idx] for
                        idx in range(cf_mat.shape[0])])
    ax.set_xticks(range(cf_mat.shape[0]))
    ax.set_xticklabels(["%s (true)" % class_label_mapping[idx] for
                        idx in range(cf_mat.shape[0])])
    plt.show()

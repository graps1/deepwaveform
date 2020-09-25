import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_rgba

from deepwaveform import waveform2matrix


def plot_waveforms(df,
                   axis,
                   classcol="class",
                   class_label_mapping=["Land", "Water"],
                   class_style_mapping=["g-", "b--"],
                   wv_cols=list(range(64))):
    """Plots the waveforms of a dataframe.

    :param df: The dataframe containing the waveforms.
    :type df: pandas.DataFrame
    :param axis: The matplotlib-axis the waveforms should be plotted on
    :type axis: matplotlib.axes.Axes
    :param classcol: Column containing class of waveform, defaults to "class"
    :type classcol: str, optional
    :param class_label_mapping: List linking class index and class names,
        defaults to ["Land", "Water"]
    :type class_label_mapping: list, optional
    :param class_style_mapping: List linking class index and waveform style,
        defaults to ["g-", "b--"]
    :type class_style_mapping: list, optional
    :param wv_cols: Columns containing waveforms, defaults to list(range(64))
    :type wv_cols: List, optional
    """
    # get waveforms in matrix form
    waveforms = waveform2matrix(df, wv_cols=wv_cols)
    # set up figure
    axis.set_title("Waveforms")
    axis.set_xlabel("Index")
    axis.set_ylabel("Amplitude")
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
        # plot full (colored) waveform (stripped of 0-elements)
        axis.plot(xs[:first_0_idx],
                  waveforms[idx, :first_0_idx],
                  class_style_mapping[dataclass],
                  label=label)
    axis.legend(loc='upper left')


def plot_pcl(df,
             axis,
             targetcol="class",
             colormap=cm.coolwarm,
             xcol="x",
             ycol="y",
             zcol="z"):
    """Plots a pointcloud from a dataframe.

    :param df: dataframe containing points
    :type df: pandas.DataFrame
    :param axis: The matplotlib-axis the pointcloud should be plotted on
    :type axis: matplotlib.axes.Axes
    :param targetcol: Column containing class of waveform, defaults to "class"
    :type targetcol: str, optional
    :param colormap: Colormap mapping class index to color,
        defaults to cm.coolwarm
    :type colormap: matplotlib.colors.Colormap, optional
    :param xcol: Column containing x-position, defaults to "x"
    :type xcol: str, optional
    :param ycol: Column containing y-position, defaults to "y"
    :type ycol: str, optional
    :param zcol: Column containing z-position, defaults to "z"
    :type zcol: str, optional
    """
    # initialize 3d-plot
    norm = Normalize(min(df[targetcol]), max(df[targetcol]))

    scatter = axis.scatter(df[xcol], df[ycol], df[zcol],
                           c=df[targetcol], cmap=colormap, norm=norm,
                           marker=".", s=0.7)
    legend = axis.legend(*scatter.legend_elements(),
                         loc="lower left", title="Classes")
    axis.add_artist(legend)
    axis.grid(False)
    axis.set_axis_off()
    axis.set_xlim3d(-60, 60)
    axis.set_ylim3d(-60, 60)
    axis.set_zlim3d(-60, 60)
    axis.view_init(elev=40., azim=-120.)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')


def plot_pcl_prediction(df,
                        axis,
                        probabilities_col=["Land", "Water"],
                        colors=["green", "blue"],
                        xcol="x",
                        ycol="y",
                        zcol="z"):
    """Plots the pointcloud of a dataframe with interpolated coloring,
    dependent on probability estimates of corresponding classes.

    :param df: dataframe containing points and probability estimates.
    :type df: pandas.DataFrame
    :param axis: The matplotlib-axis the pointcloud should be plotted on
    :type axis: matplotlib.axes.Axes
    :param probabilities_col: Columns with probability estimates,
        defaults to ["Land", "Water"]
    :type probabilities_col: list, optional
    :param colors: Colors associates with probability estimates,
        defaults to ["green", "blue"]
    :type colors: list, optional
    :param xcol: Column containing x-position, defaults to "x"
    :type xcol: str, optional
    :param ycol: Column containing y-position, defaults to "y"
    :type ycol: str, optional
    :param zcol: Column containing z-position, defaults to "z"
    :type zcol: str, optional
    """
    probarr = np.array(df[probabilities_col])
    # linear interpolation between colors
    rgba_colors = np.array([to_rgba(color) for color in colors])
    colorsarr = np.zeros(shape=(df.shape[0], 4))
    for idx, col in enumerate(rgba_colors):
        colorsarr += (probarr[:, idx]*np.array([col]*df.shape[0]).T).T
    colorsarr = (colorsarr - np.min(colorsarr))/(
        np.max(colorsarr) - np.min(colorsarr))

    # initialize 3d-plot
    axis.scatter(df[xcol], df[ycol], df[zcol],
                 color=colorsarr,
                 marker=".", s=0.7)

    for color, classname in zip(colors, probabilities_col):
        axis.add_line(plt.Line2D([0], [0], color=color, label=classname, lw=4))

    legend = axis.legend(loc="lower left", title="Classes")
    axis.add_artist(legend)

    axis.grid(False)
    axis.set_axis_off()
    axis.set_xlim3d(-60, 60)
    axis.set_ylim3d(-60, 60)
    axis.set_zlim3d(-60, 60)
    axis.view_init(elev=40., azim=-120.)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')


def plot_training_progress(stats, axis):
    """Plots the training process of a network over time.

    :param stats: List of dictionaries, containing entries for
        "meanloss" and "varloss".
    :type stats: List[Dict]
    :param axis: The matplotlib-axis the training process should be
        plotted on.
    :type axis: matplotlib.axes.Axes
    """
    xs = np.arange(len(stats))
    ys = np.array([stat["meanloss"] for stat in stats])
    ys_sd = np.array([stat["varloss"] for stat in stats])**0.5
    axis.plot(xs, ys, color="red")
    axis.fill_between(xs, ys-0.5*ys_sd, ys+0.5*ys_sd, color="red", alpha=0.2)
    axis.set_xlabel("epoch")
    axis.set_ylabel("mean loss")


def plot_confusion_matrix(model,
                          axis,
                          dataset,
                          class_label_mapping=["Land", "Water"]):
    """Plots the confusion matrix of a classifier w.r.t. a dataset.

    :param model: The classifier
    :type model: deepwaveform.ConvNet
    :param axis: The matplotlib-axis the confusion matrix should be plotted on
    :type axis: matplotlib.axes.Axes
    :param dataset: The dataset
    :type dataset: deepwaveform.WaveFormDataset
    :param class_label_mapping: List linking class indices to class names,
        defaults to ["Land", "Water"]
    :type class_label_mapping: list, optional
    """
    wf, lab = dataset[:]["waveform"], dataset[:]["label"].numpy()
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
            axis.text(j, i, "%.3f" % p, va="center", ha="center")

    axis.matshow(cf_mat, cmap=cm.coolwarm)
    axis.set_yticks(range(cf_mat.shape[0]))
    axis.set_yticklabels(["%s (predicted)" % class_label_mapping[idx] for
                         idx in range(cf_mat.shape[0])])
    axis.set_xticks(range(cf_mat.shape[0]))
    axis.set_xticklabels(["%s (true)" % class_label_mapping[idx] for
                         idx in range(cf_mat.shape[0])])

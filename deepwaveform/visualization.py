import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_rgba
import plotly.graph_objects as go

from deepwaveform import waveform2matrix


def plot_waveforms(df,
                   axis,
                   classcol="class",
                   class_label_mapping=None,
                   class_style_mapping=None,
                   wv_cols=list(map(str, range(64)))):
    """Plots the waveforms of a dataframe.

    :param df: The dataframe containing the waveforms.
    :type df: pandas.DataFrame
    :param axis: The matplotlib-axis the waveforms should be plotted on
    :type axis: matplotlib.axes.Axes
    :param classcol: Column containing class of waveform, defaults to "class"
    :type classcol: str, optional
    :param class_label_mapping: List linking class index and class names,
        defaults to None
    :type class_label_mapping: list[str], optional
    :param class_style_mapping: List linking class index and waveform style,
        defaults to None
    :type class_style_mapping: list[str], optional
    :param wv_cols: Columns containing waveforms,
        defaults to list(map(str, range(64)))
    :type wv_cols: List, optional
    """
    if class_label_mapping is None:
        class_label_mapping = list(map(str, range(max(df[classcol])+1)))
    if class_style_mapping is None:
        class_style_mapping = [
            "b-", "g--", "r-", "c--", "m-", "y--",
            "k-", "b--", "g-", "r--"][:len(class_label_mapping)]
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
             class_label_mapping=None,
             colors=None,
             xcol="x",
             ycol="y",
             zcol="z",
             use_plotly=False,
             inv_z=True):
    """Plots a pointcloud from a dataframe.

    :param df: dataframe containing points
    :type df: pandas.DataFrame
    :param axis: The matplotlib-axis the pointcloud should be plotted on
    :type axis: matplotlib.axes.Axes
    :param targetcol: Column containing class of waveform, defaults to "class"
    :type targetcol: str, optional
    :param class_label_mapping: List linking class index and class names,
        defaults to None
    :type class_label_mapping: list[str], optional
    :param colors: Colors mapping class index to color,
        defaults to None
    :type colors: list[str], optional
    :param xcol: Column containing x-position, defaults to "x"
    :type xcol: str, optional
    :param ycol: Column containing y-position, defaults to "y"
    :type ycol: str, optional
    :param zcol: Column containing z-position, defaults to "z"
    :type zcol: str, optional
    :param use_plotly: Whether plotly should be used for plotting or not
    :type use_plotly: bool, optional
    :param inv_z: Whether the z-axis should be inverted.
    :type inv_z: bool, optional
    """
    if class_label_mapping is None:
        class_label_mapping = list(map(str, range(max(df[targetcol])+1)))
    if colors is None:
        colors = ["blue", "orange", "green", "red", "purple", "brown", 
                  "pink", "gray", "olive", "cyan"][:len(class_label_mapping)]

    if use_plotly:
        traces = []
        for idx, classname in enumerate(class_label_mapping):
            selected = df[df[targetcol] == idx]
            trace = go.Scatter3d(
                x=selected[xcol], 
                y=selected[ycol], 
                z=(-1 if inv_z else 1)*selected[zcol], 
                opacity=1,
                name=classname,
                mode='markers',
                marker=dict(
                    size=1,
                    color=colors[idx],
                    opacity=1
                )
            )
            traces.append(trace)
        layout = go.Layout(
            scene=dict(aspectmode="data")
        )
        fig = go.Figure(data=traces, layout=layout)
        fig.show()
    else:
        # initialize 3d-plot
        norm = Normalize(min(df[targetcol]), max(df[targetcol]))
        for idx, classname in enumerate(class_label_mapping):
            selected = df[df[targetcol] == idx]
            axis.scatter(selected[xcol], 
                         selected[ycol],
                         (-1 if inv_z else 1)*selected[zcol],
                         color=colors[idx],
                         label=classname,
                         norm=norm,
                         marker=".",
                         s=0.7)
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


def plot_pcl_prediction(df,
                        axis,
                        probabilities_col=None,
                        colors=None,
                        xcol="x",
                        ycol="y",
                        zcol="z",
                        use_plotly=False,
                        inv_z=True):
    """Plots the pointcloud of a dataframe with interpolated coloring,
    dependent on probability estimates of corresponding classes.

    :param df: dataframe containing points and probability estimates.
    :type df: pandas.DataFrame
    :param axis: The matplotlib-axis the pointcloud should be plotted on
    :type axis: matplotlib.axes.Axes
    :param probabilities_col: Columns with probability estimates,
        defaults to None
    :type probabilities_col: list, optional
    :param colors: Colors associates with probability estimates,
        defaults to None
    :type colors: list, optional
    :param xcol: Column containing x-position, defaults to "x"
    :type xcol: str, optional
    :param ycol: Column containing y-position, defaults to "y"
    :type ycol: str, optional
    :param zcol: Column containing z-position, defaults to "z"
    :type zcol: str, optional
    :param use_plotly: Whether plotly should be used for plotting or not
    :type use_plotly: bool, optional
    :param inv_z: Whether the z-axis should be inverted.
    :type inv_z: bool, optional
    """
    if probabilities_col is None:
        probabilities_col = [col for col in df.columns if "pred_" in col]
    if colors is None:
        colors = ["blue", "orange", "green", "red", "purple", "brown",
                  "pink", "gray", "olive", "cyan"][:len(probabilities_col)]

    probarr = np.array(df[probabilities_col])
    # linear interpolation between colors
    rgba_colors = np.array([to_rgba(color) for color in colors])
    colorsarr = np.zeros(shape=(df.shape[0], 4))
    for idx, col in enumerate(rgba_colors):
        colorsarr += (probarr[:, idx]*np.array([col]*df.shape[0]).T).T
    colorsarr = (colorsarr - np.min(colorsarr))/(
        np.max(colorsarr) - np.min(colorsarr))

    if use_plotly:
        layout = go.Layout(
            scene=dict(aspectmode="data"),
            showlegend=True
        )
        fig = go.Figure(layout=layout)
        predicted = np.argmax(df[probabilities_col].to_numpy(), axis=1)
        for idx, colname in enumerate(probabilities_col):
            selected = df[predicted == idx]
            selectedcolors = colorsarr[predicted == idx, :]
            fig.add_trace(go.Scatter3d(
                x=selected[xcol], 
                y=selected[ycol], 
                z=(-1 if inv_z else 1)*selected[zcol], 
                opacity=1,
                mode='markers',
                marker=dict(
                    size=1,
                    color=selectedcolors,
                    opacity=1
                ),
                name=colname
            ))

        fig.show()
    else:
        # initialize 3d-plot
        axis.scatter(df[xcol], df[ycol], (-1 if inv_z else 1)*df[zcol],
                     color=colorsarr,
                     marker=".", s=0.7)

        for color, classname in zip(colors, probabilities_col):
            axis.add_line(plt.Line2D(
                [0], [0], color=color, label=classname, lw=4))

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

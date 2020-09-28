import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import nn

import numpy as np

from deepwaveform import waveform2matrix


class ConvNet(nn.Module):
    def __init__(self,
                 input_dimension=64,
                 output_dimension=2,
                 layer1=(5, 9),
                 layer2=(5, 15)):
        """Initializes a simple one dimensional CNN with two layers.

        :param input_dimension: Length of a waveform, defaults to 64
        :type input_dimension: int, optional
        :param output_dimension: Number of classes, defaults to 2
        :type output_dimension: int, optional
        :param layer1: (number kernels of first layer, kernel sizes),
            defaults to (5, 9)
        :type layer1: tuple, optional
        :param layer2: (number of kernels of second layer, kernel sizes),
            defaults to (5, 15)
        :type layer2: tuple, optional
        """
        assert layer1[1] % 2 != 0, "kernel size must be uneven"
        assert layer2[1] % 2 != 0, "kernel size must be uneven"
        assert input_dimension % 4 == 0, "input dimension not divisible by 4"

        super(ConvNet, self).__init__()
        self.output_dimension = output_dimension

        self.features = nn.Sequential(
            nn.Conv1d(1, layer1[0],
                      kernel_size=layer1[1],
                      padding=int(layer1[1]/2)),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(layer1[0], layer2[0],
                      kernel_size=layer2[1],
                      padding=int(layer2[1]/2)),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(layer2[0]*(input_dimension//4), output_dimension),
        )

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def predict(self, x):
        return F.softmax(self(x).data, dim=1).numpy()

    def annotate_dataframe(self, df, wv_cols=list(map(str, range(64))),
                           class_label_mapping=None,
                           predicted_column="predicted"):
        """Annotates a dataframe containing waveforms with the estimated
        classes and respective probability estimates.

        :param df: Dataframe with waveforms
        :type df: pandas.DataFrame
        :param wv_cols: List of column names containing the waveforms,
            defaults to list(map(str, range(64)))
        :type wv_cols: list, optional
        :param class_label_mapping: New columns for probability estimates,
            defaults to None
        :type class_label_mapping: list, optional
        :param predicted_column: New column for class estimates,
            defaults to "predicted"
        :type predicted_column: str, optional
        """
        if class_label_mapping is None:
            class_label_mapping = [
                "pred_%d" % idx for idx in range(self.output_dimension)]

        # adds new columns to dataframe
        ds = WaveFormDataset(df,
                             classcol=None,
                             wv_cols=wv_cols)
        pred = self.predict(ds[:]["waveform"])
        df[predicted_column] = np.argmax(pred, axis=1)
        for idx, label in enumerate(class_label_mapping):
            df[label] = pred[:, idx]


class AutoEncoder(nn.Module):
    def __init__(self, layer1=64, layer2=32, hidden=5):
        """A simple symmetric Autoencoder with 2 encoding layers,
        one hidden and 2 decoding layers.

        :param layer1: Dimension of first/fifth layer, defaults to 64
        :type layer1: int, optional
        :param layer2: Dimension of second/fourth layer, defaults to 32
        :type layer2: int, optional
        :param hidden: Dimension of hidden layer, defaults to 5
        :type hidden: int, optional
        """
        super(AutoEncoder, self).__init__()
        self.hiddensize = hidden

        self.encode = nn.Sequential(
            nn.Linear(layer1, layer2),
            nn.ReLU(inplace=True),
            nn.Linear(layer2, hidden),
            nn.ReLU(inplace=True)
        )

        self.decode = nn.Sequential(
            nn.Linear(hidden, layer2),
            nn.ReLU(inplace=True),
            nn.Linear(layer2, layer1)
        )

    def encoder(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        out = self.encode(x)
        return out

    def decoder(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        out = self.decode(x)
        return out

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

    def annotate_dataframe(self, df, wv_cols=list(map(str, range(64))),
                           encoding_prefix="hidden_",
                           reconstruction_prefix="reconstr_"):
        """Annotates a dataframe with the estimated encoding and
        the corresponding reconstruction.

        :param df: Dataframe with waveforms
        :type df: pandas.DataFrame
        :param wv_cols: List of column names containing waveforms,
            defaults to list(map(str, range(64)))
        :type wv_cols: list, optional
        :param encoding_prefix: Prefix of new columns containing encoding,
            defaults to "hidden\_"
        :type encoding_prefix: str, optional
        :param reconstruction_prefix: Prefix of new columns containing
            reconstruction, defaults to "reconstr\_"
        :type reconstruction_prefix: str, optional
        """
        ds = WaveFormDataset(df, classcol=None,
                             wv_cols=list(map(str, range(64))))
        hidden = self.encoder(ds[:]["waveform"])
        output = self.decoder(hidden).detach().numpy()
        hidden = hidden.detach().numpy()
        # denormalize
        output = output*(ds.ma-ds.mi)+ds.mi
        for idx in range(self.hiddensize):
            df["%s%d" % (encoding_prefix, idx)] = hidden[:, 0, idx]
        for idx, wv in enumerate(wv_cols):
            df["%s%s" % (reconstruction_prefix, wv)] = output[:, 0, idx]


class WaveFormDataset(Dataset):

    def __init__(self,
                 df,
                 classcol="class",
                 wv_cols=list(map(str, range(64)))):
        """Initializes a dataset that can be processed by neural networks.

        :param df: The dataframe containing the waveforms
        :type df: pandas.DataFrame
        :param classcol: Column in dataframe containing class of waveform.
            If `None` then class of waveform is ignored -- e.g. when it comes
            to autoencoders, defaults to "class"
        :type classcol: str, optional
        :param wv_cols: Columns dataframe containing waveforms,
            defaults to list(map(str, range(64)))
        :type wv_cols: list, optional
        """
        super(WaveFormDataset, self).__init__()
        datamatrix = waveform2matrix(df, wv_cols=wv_cols)
        self.mi, self.ma = np.min(datamatrix), np.max(datamatrix)
        self.xs = (datamatrix - self.mi)/(self.ma-self.mi)
        self.xs = torch.tensor(self.xs).float()
        self.labels = torch.tensor(df[classcol].to_numpy()) if \
            classcol is not None else [0]*len(self.xs)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {"waveform": self.xs[idx], "label": self.labels[idx]}
        return sample


class Trainer:
    def __init__(self,
                 model,
                 dataset,
                 optimizer=None,
                 batch_size=128,
                 epochs=20):
        """Initializes a trainer for a neural network.

        :param model: The model
        :type model: deepwaveform.ConvNet or deepwaveform.AutoEncoder
        :param dataset: The dataset the model should be trained on
        :type dataset: deepwaveform.WaveFormDataset
        :param optimizer: The optimizer that should be used. If None,
            then Adam is used, defaults to None
        :type optimizer: torch.optim.Optimizer, optional
        :param batch_size: The batch size used for training, defaults to 128
        :type batch_size: int, optional
        :param epochs: The number of epochs used for training, defaults to 20
        :type epochs: int, optional
        """
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset = dataset

        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                weight_decay=0)

    def _train(self, criterion):

        train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True)

        for epoch in range(self.epochs):
            sum_xi, sum_xi_squared = 0, 0
            meanloss, varloss = 0, 0
            for i, d in enumerate(train_loader):
                waveforms, labels = d["waveform"], d["label"]

                waveforms.requires_grad_(True)
                loss = criterion(waveforms, labels)
                waveforms.retain_grad()
                waveforms.requires_grad_(False)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                sum_xi += loss.item()
                sum_xi_squared += loss.item()**2
                varloss = max(0, 1/(i+1)*(sum_xi_squared - sum_xi/(i+1)))
                meanloss = meanloss + (loss.item() - meanloss)/(i+1)

            result = {"meanloss": meanloss, "varloss": varloss}
            yield result

    def train_classifier(self):
        """Trains a classifier over the dataset.

        :return: An iterator over the results of each epoch. Each
            result is given as a dictionary with keys "meanloss"
            and "varloss"
        :rtype: Iterator[Dict]
        """
        cel = nn.CrossEntropyLoss()

        def criterion(wfs, labels):
            return cel(self.model(wfs), labels)

        return self._train(criterion)

    def train_autoencoder(self, sparsity=0):
        """Trains an autoencoder over the dataset.

        :param sparsity: multiplier for l1 penalty on the hidden layer,
            defaults to 0
        :type sparsity: int, optional
        :return: An iterator over the results of each epoch. Each
            result is given as a dictionary with keys "meanloss"
            and "varloss"
        :rtype: Iterator[Dict]
        """
        msel = nn.MSELoss()
        l1l = nn.L1Loss()

        def criterion(wfs, labels):
            if len(wfs.size()) == 2:
                wfs = wfs.unsqueeze(1)
            hidden = self.model.encoder(wfs)
            modeloutput = self.model.decoder(hidden)
            return msel(modeloutput, wfs) +\
                sparsity*l1l(hidden, torch.zeros(size=hidden.size()))

        return self._train(criterion)

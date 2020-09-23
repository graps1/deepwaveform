import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn

import numpy as np

from deepwaveform import waveform2matrix


class ConvNet(nn.Module):
    def __init__(self, output_dimension=2):
        super(ConvNet, self).__init__()
        self.output_dimension = output_dimension

        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(8*16, output_dimension),
        )
        
    def forward(self, x):
        out = self.features(x.unsqueeze(1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def predict(self, x):
        return F.softmax(self(x).data, dim=1).numpy()

    def process_dataframe(self, df, wv_cols=list(range(64)),
                          class_label_mapping=["Land", "Water"],
                          predicted_column="Predicted"):
        # adds new columns to dataframe
        ds = WaveFormDataset(df,
                             classcol=None,
                             wv_cols=wv_cols)
        pred = self.predict(ds[:]["waveform"])
        df[predicted_column] = np.argmax(pred, axis=1)
        for idx, label in enumerate(class_label_mapping):
            df[label] = pred[:, idx]


class AutoEncoder(nn.Module):
    def __init__(self, dimensions=5):
        super(AutoEncoder, self).__init__()
        self.dimensions = dimensions

        self.encode = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True)
        )

        self.h = nn.Sequential(
            nn.Linear(16, dimensions),
            nn.ReLU(inplace=True)
        )

        self.decode = nn.Sequential(
            nn.Linear(dimensions, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Sequential(
            nn.Linear(32, 64)
        )

    def encoder(self, x):
        out = self.encode(x.unsqueeze(1))
        out = out.view(out.size(0), -1)
        out = self.h(out)
        return out

    def decoder(self, x):
        out = x.view(x.size(0), 1, x.size(1))
        out = self.decode(out)
        out = out.view(out.size(0), -1)
        out = self.output(out)
        return out

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


class WaveFormDataset(Dataset):
    def __init__(self,
                 df,
                 classcol="class",
                 wv_cols=list(range(64))):
        super(WaveFormDataset, self).__init__()
        datamatrix = waveform2matrix(df, wv_cols=wv_cols)
        mi, ma = np.min(datamatrix), np.max(datamatrix)
        self.xs = (datamatrix - mi)/(ma-mi)         # Normalize
        self.xs = torch.tensor(self.xs).float()     # to tensor
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
                varloss = 1/(i+1)*(sum_xi_squared - sum_xi/(i+1))
                meanloss = meanloss + (loss.item() - meanloss)/(i+1)

            result = {"meanloss": meanloss, "varloss": varloss}
            yield result

    def train_classifier(self):
        cel = nn.CrossEntropyLoss()

        def criterion(wfs, labels):
            return cel(self.model(wfs), labels)

        return self._train(criterion)

    def train_autoencoder(self, sparsity=0):
        msel = nn.MSELoss()
        l1l = nn.L1Loss()

        def criterion(wfs, labels):
            hidden = self.model.encoder(wfs)
            modeloutput = self.model.decoder(hidden)
            return msel(modeloutput, wfs) + sparsity*l1l(hidden, 0)

        return self._train(criterion)

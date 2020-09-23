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


class WaveFormDataset(Dataset):
    def __init__(self,
                 df,
                 classcol="class",
                 cutleft=0,
                 cutright=64,
                 wv_dim=200):
        super(WaveFormDataset, self).__init__()
        datamatrix = waveform2matrix(df, wv_dim=wv_dim)[:, cutleft:cutright]
        mi, ma = np.min(datamatrix), np.max(datamatrix)
        self.xs = (datamatrix - mi)/(ma-mi)         # Normalize
        self.xs = torch.tensor(self.xs).float()     # to tensor
        self.labels = torch.tensor(df[classcol].to_numpy())

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
                 ptrain=0.9,
                 batch_size=128,
                 epochs=20):
        self.model = model
        self.ptrain = ptrain
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset = dataset

        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                weight_decay=0)

    def _train(self, criterion):

        train_size = int(len(self.dataset)*self.ptrain)
        train_dataset, test_dataset = random_split(
            self.dataset,
            [train_size, len(self.dataset) - train_size])

        train_loader = DataLoader(
            dataset=train_dataset,
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
        loss = nn.CrossEntropyLoss()

        def criterion(wfs, labels):
            return loss(self.model(wfs), labels)

        return self._train(criterion)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn

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
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def predict(self, x):
        return F.softmax(self(x).data, dim=1).numpy()

class WaveFormDataset(Dataset):
    def __init__(self, df, class_col="class", cutoff=64, wv_dim=200):
        self.xs = waveform2matrix(df,wv_dim=wv_dim)[:,:cutoff]
        self.xs = (self.xs - np.min(self.xs))/(np.max(self.xs)-np.min(self.xs)) # Normalize
        self.xs = torch.tensor(self.xs).float() # to tensor
        self.labels = torch.tensor(data[class_col].to_numpy())

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = { "waveform" : self.xs[idx], "label" : self.labels[idx] }
        return sample
import torch
from torch import nn
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self, output_dimension=2):
        """A simple one dimensional convolutional neural network.

        :param output_dimension: Output dimension, i.e. the number of classes. 
            Defaults to 2.
        :type output_dimension: int, optional
        """

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
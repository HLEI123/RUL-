import torch
from torch import nn
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        # [b, 784] => [b, 20]
        self.encoder = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        # [b, 20] => [b, 784]
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1280),
            # nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: [b, 1, 28, 28]
        :return:
        """
        batch_size = x.size(0)
        # flatten
        x = x.view(batch_size, 1280)
        # encoder
        x = self.encoder(x)
        # decoder
        x = self.decoder(x)
        # reshape
        x = x.view(batch_size,1280)

        return x, None


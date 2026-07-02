from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphNeuralNet(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

        self.train_loss = []
        self.val_loss = []

        self.train_f1 = []
        self.val_f1 = []

        self.conv1 = SAGEConv(2, 32)
        self.conv2 = SAGEConv(32, 32)
        self.conv3 = SAGEConv(32, 32)

        self.edge_mlp = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        src, dst = edge_index

        edge_feat = torch.cat([x[src], x[dst]], dim=1)

        return self.edge_mlp(edge_feat).squeeze(-1)

    def save(self):
        """
        Short function that allows saving of the current state

        Args:
            filepath: pretty self explanatory, refers to the directory
            filename: pretty self explanatory, refers to the actual filename with extention
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "train_f1": self.train_f1,
                "val_f1": self.val_f1,
            },
            Path("models") / f"{self.name}.pt",
        )

    def load(self, device="cpu"):
        # def load(self, filepath: Path = None, device='cpu'):
        """
        Short function that allows reloading of weights from a savefile.

        Args:
            filepath: pretty self explanatory, refers to the directory
            Device: loads instantly to the gpu/cpu
        """
        checkpoint = torch.load(Path("models") / f"{self.name}.pt", map_location=device)

        self.load_state_dict(checkpoint["state_dict"])
        self.train_loss = checkpoint["train_loss"]
        self.val_loss = checkpoint["val_loss"]
        self.train_f1 = checkpoint["train_f1"]
        self.val_f1 = checkpoint["val_f1"]

    def plot_performance(self):
        epochs = np.arange(1, len(self.train_loss) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        axes[0].plot(epochs, self.train_loss, label="Train")
        axes[0].plot(epochs, self.val_loss, label="Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("BCE Loss")
        axes[0].set_title("Loss troughout training")
        axes[0].legend()
        axes[0].grid(True)

        # F1-score
        axes[1].plot(epochs, self.train_f1, label="Train")
        axes[1].plot(epochs, self.val_f1, label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1-score")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("F1-score troughout training")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(Path("models") / f"{self.name}_training.png")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class GraphNeuralNet(nn.Module):
    """
    GNN for hit-to-track association via edge classification.

    Three SAGEConv message-passing layers build node embeddings from raw
    (x, y) hit coordinates. An edge MLP then scores each candidate edge by
    concatenating the embeddings of its two endpoints and predicting whether
    they belong to the same track.

    The residual connection (x1 passed into conv3 via x2) preserves early
    geometric information after several rounds of neighbourhood aggregation,
    which helps when hits from different tracks overlap spatially.

    Parameters
    ----------
    name : str
        Used as the filename stem for saving checkpoints and plots
        (saved to models/<name>.pt and models/<name>_training.png).
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.train_loss: list = []
        self.val_loss: list = []
        self.train_f1: list = []
        self.val_f1: list = []
        self.conv1 = SAGEConv(3, 64)
        self.conv2 = SAGEConv(64, 64)
        self.conv3 = SAGEConv(64, 64)
        self.edge_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, graph: Data) -> torch.Tensor:
        """
        Forward pass: message passing followed by edge scoring.

        Parameters
        ----------
        graph : torch_geometric.data.Data
            A single graph or a PyG batch. Expected attributes:
              graph.x          — node features, shape (n_hits, 3)
              graph.edge_index — candidate edges, shape (2, n_edges)

        Returns
        -------
        torch.Tensor, shape (n_edges,)
            Raw logits, one per candidate edge. Apply sigmoid for
            P(same track), or pass directly to BCEWithLogitsLoss.
        """
        x, edge_index = graph.x, graph.edge_index
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = self.conv3(x2, edge_index)
        src, dst = edge_index  # pyright: ignore[reportGeneralTypeIssues]
        return self.edge_mlp(torch.cat([x3[src], x3[dst]], dim=1)).squeeze(-1)

    def save(self) -> None:
        """
        Save model weights and training history to models/<name>.pt.

        Only call this when validation loss has improved so the checkpoint
        always reflects the best observed weights.
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

    def load(self, device: str = "cpu") -> None:
        """
        Restore weights and training history from models/<name>.pt.

        Parameters
        ----------
        device : str, optional (default="cpu")
            Device to map the checkpoint onto. Pass "cuda" to load
            directly onto GPU without an intermediate CPU copy.
        """
        checkpoint = torch.load(
            Path("models") / f"{self.name}.pt", map_location=device, weights_only=False
        )
        self.load_state_dict(checkpoint["state_dict"])
        self.train_loss = checkpoint["train_loss"]
        self.val_loss = checkpoint["val_loss"]
        self.train_f1 = checkpoint["train_f1"]
        self.val_f1 = checkpoint["val_f1"]

    def plot_performance(self) -> None:
        """
        Plot loss and F1-score curves and save to models/<name>_training.png.

        Both train and validation curves are shown side by side so
        overfitting is immediately visible as a divergence between the two.
        """
        epochs = np.arange(1, len(self.train_loss) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, self.train_loss, label="Train")
        axes[0].plot(epochs, self.val_loss, label="Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("BCE loss")
        axes[0].set_title("Loss throughout training")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(epochs, self.train_f1, label="Train")
        axes[1].plot(epochs, self.val_f1, label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1-score")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("F1-score throughout training")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(Path("models") / f"{self.name}_training.png")

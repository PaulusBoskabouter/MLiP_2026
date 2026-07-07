from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from objects.collider import CollisionDetector2D, TrackHit
from objects.network import GraphNeuralNet
from objects.utilities import (
    HitDataset,
    export_predictions,
    hit_accuracy,
    test_model,
    train_model,
    visual_example,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


model = GraphNeuralNet("GNN_linear_tracks")
model.load()
model.plot_performance()


model = GraphNeuralNet("GNN_curved_tracks")
model.load()
model.plot_performance()

import torch
from objects.network import GraphNeuralNet

device = "cuda" if torch.cuda.is_available() else "cpu"


straight_model = GraphNeuralNet("GNN_linear_tracks")
straight_model.load(device=device)
straight_model.plot_performance()


curved_model = GraphNeuralNet("GNN_curved_tracks")
curved_model.load(device=device)
curved_model.plot_performance()

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from objects.graph import build_graph


class HitDataset(Dataset):
    def __init__(self, events):
        self.events = events

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        hits = self.events[idx]
        node_features, edge_index, edge_labels = build_graph(hits)

        return Data(x=node_features, edge_index=edge_index, edge_label=edge_labels)


def train_model(
    model, train_loader, val_loader, epochs=100, lr=0.001, device="cpu", patience=15
):
    """
    Main training loop of the model
    Parameters
    ----------
    Model: object.NeuralNet object.
        Name says it all, it's the graph neural network
    ...
    """
    # Make sure the log file is empty
    with open(Path("models") / f"{model.name}.txt", "w") as file:
        file.write("")
        file.close()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    model = model.to(device)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        # Training
        model.train()
        running_loss = 0.0
        running_f1 = 0.0

        for graph in train_loader:
            graph = graph.to(device)

            optimizer.zero_grad()
            pred = model(graph)

            # Determine accuracy
            probs = torch.sigmoid(pred)
            pred_labels = (probs >= 0.5).float()

            y_true = graph.edge_label.view(-1).detach().cpu().numpy()
            y_pred = pred_labels.view(-1).detach().cpu().numpy()

            running_f1 += f1_score(y_true, y_pred)

            # Back propagation
            loss = criterion(pred, graph.edge_label.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_f1 = running_f1 / len(train_loader)
        model.train_loss.append(train_loss)
        model.train_f1.append(train_f1)

        # Validation
        model.eval()
        running_loss = 0.0
        running_f1 = 0.0

        with torch.no_grad():
            for graph in val_loader:
                graph = graph.to(device)

                pred = model(graph)

                # Determine f1 score
                probs = torch.sigmoid(pred)
                pred_labels = (probs >= 0.5).float()
                y_true = graph.edge_label.view(-1).detach().cpu().numpy()
                y_pred = pred_labels.view(-1).detach().cpu().numpy()
                running_f1 += f1_score(y_true, y_pred)

                # Determine loss
                loss = criterion(pred, graph.edge_label.float())
                running_loss += loss.item()

            val_loss = running_loss / len(val_loader)
            val_f1 = running_f1 / len(val_loader)

            model.val_loss.append(val_loss)
            model.val_f1.append(val_f1)

        # Validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model.save()
        elif epochs_no_improve >= patience:
            break
        else:
            epochs_no_improve += 1

        logs = f"Epoch [{epoch + 1:3d}/{epochs}] | Train loss: {train_loss:.5f} | Val loss {val_loss:.5f} | Best loss: {best_val_loss:.5f}| Train f1 {train_f1:.5f} | Val f1 {val_f1:.5f} | Patience: [{epochs_no_improve:2d}/{patience}]\n"
        with open(Path("models") / f"{model.name}.txt", "a") as file:
            file.write(logs)
            file.close()
        pbar.set_postfix(
            train_loss=f"{train_loss:.5f}",
            val_loss=f"{val_loss:.5f}",
            best_loss="{best_val_loss:.5f}",
            patience=f"[{epochs_no_improve:2d}/{patience}]",
        )

    model.plot_performance()


def test_model(model, test_loader, device="cpu"):
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()

    running_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for graph in test_loader:
            graph = graph.to(device)

            pred = model(graph)
            loss = criterion(pred, graph.edge_label.float())
            running_loss += loss.item()

            probs = torch.sigmoid(pred)
            pred_labels = (probs >= 0.5).float()

            all_preds.append(pred_labels.cpu())
            all_targets.append(graph.edge_label.cpu())

    # ---------- metrics ----------
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    loss = running_loss / len(test_loader)
    f1 = f1_score(all_targets, all_preds)

    return loss, f1


def get_predicted_clusters(edge_index, pred_edges, num_nodes):
    G = nx.Graph()

    G.add_nodes_from(range(num_nodes))

    src, dst = edge_index
    for i in range(len(pred_edges)):
        if pred_edges[i]:
            G.add_edge(int(src[i]), int(dst[i]))

    clusters = list(nx.connected_components(G))

    node_to_cluster = np.zeros(num_nodes, dtype=int)

    for cluster_id, nodes in enumerate(clusters):
        for n in nodes:
            node_to_cluster[n] = cluster_id

    return node_to_cluster


def visual_example(model, graph, sample_hits, radii, device="cpu"):
    graph = next(iter(graph))
    hits = sample_hits[0]

    graph = graph.to(device)

    logits = model(graph)
    probs = torch.sigmoid(logits)
    pred_edges = (probs > 0.5).cpu().numpy()

    pred_clusters = get_predicted_clusters(
        graph.edge_index.cpu().numpy(), pred_edges, graph.x.size(0)
    )

    true_clusters = np.array([h.track_id for h in hits])

    plot_comparison(
        hits, true_clusters, pred_clusters, radii, f"{model.name}_visual_sample.png"
    )


def plot_comparison(hits, true_clusters, pred_clusters, radii, file_name):
    pos = np.array([(h.x, h.y) for h in hits])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax in axes:
        # draw circles on BOTH axes
        for r in radii:
            circle = plt.Circle((0, 0), r, fill=False, color="green", alpha=0.4)  # pyright: ignore[reportPrivateImportUsage]
            ax.add_patch(circle)

        # reference point
        ax.scatter([0], [0], c="black", marker="x")

        # axis formatting applied to BOTH
        lim = radii[-1] * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")

    # left plot: ground truth
    axes[0].scatter(pos[:, 0], pos[:, 1], c=true_clusters, cmap="tab10", s=40)
    axes[0].set_title("Ground truth")

    # right plot: prediction
    axes[1].scatter(pos[:, 0], pos[:, 1], c=pred_clusters, cmap="tab10", s=40)
    axes[1].set_title("Predicted")

    plt.tight_layout()
    plt.savefig(Path("results") / file_name)

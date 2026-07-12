from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from objects.graph import build_graph
from objects.network import GraphNeuralNet


class HitDataset(Dataset):
    """
    PyTorch Geometric dataset that converts raw events into graphs.

    Graphs are built once at construction and cached in memory, so
    build_graph only runs once per event regardless of how many epochs
    are trained. Each graph is stored as a Pytorch Geometric Data object and served
    to a DataLoader.

    Parameters
    ----------
    events : list of list of TrackHit
        One inner list per collision event, each containing all hits
        recorded in that event.
    desc : str, optional (default="Dataset")
        Label shown in the tqdm progress bar during graph construction.
    """

    def __init__(self, events: list, desc: str = "Dataset", hits: list = []):
        self.data = []
        track_ids = torch.tensor([h.track_id for h in hits], dtype=torch.long)

        for hits in tqdm(events, desc=f"Building graphs ({desc})"):
            node_features, edge_index, edge_labels = build_graph(hits)
            self.data.append(
                Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_label=edge_labels,
                    track_id=track_ids,  # ← add this
                )
            )

    def __len__(self) -> int:
        """Return the number of events in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Data:
        """
        Return the pre-built graph for event idx.

        Parameters
        ----------
        idx : int
            Index into the dataset.

        Returns
        -------
        torch_geometric.data.Data
            Graph with attributes:
              .x          — node features, shape (n_hits, 2)
              .edge_index — candidate edges, shape (2, n_edges)
              .edge_label — ground-truth labels, shape (n_edges,)
        """
        return self.data[idx]


def train_model(
    model: GraphNeuralNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 15,
) -> None:
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
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        logs = f"Epoch [{epoch + 1:3d}/{epochs}] | Train loss: {train_loss:.5f} | Val loss {val_loss:.5f} | Best loss: {best_val_loss:.5f}| Train f1 {train_f1:.5f} | Val f1 {val_f1:.5f} | Patience: [{epochs_no_improve:2d}/{patience}]\n"
        with open(Path("models") / f"{model.name}.txt", "a") as file:
            file.write(logs)
            file.close()
        pbar.set_postfix(
            train_loss=f"{train_loss:.5f}",
            val_loss=f"{val_loss:.5f}",
            best_loss=f"{best_val_loss:.5f}",
            patience=f"[{epochs_no_improve:2d}/{patience}]",
        )

    model.plot_performance()


def test_model(
    model: GraphNeuralNet, test_loader: DataLoader, device: str = "cpu"
) -> tuple[float, float]:
    """
    Evaluate the model on a held-out test set.

    Computes three metrics:
      - BCE loss: edge-level loss averaged over all batches.
      - F1 score: edge-level, comparing predicted same/different-track
        labels against ground truth.

    Parameters
    ----------
    model : GraphNeuralNet
        Trained model to evaluate.
    test_loader : DataLoader
        DataLoader wrapping the test HitDataset.
    device : str, optional (default="cpu")
        Device to run inference on.

    Returns
    -------
    loss : float
        Average BCE loss over the test set.
    f1 : float
        Edge-level F1 score over the test set.

    """
    model.eval()
    model = model.to(device)
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


def hit_accuracy(
    model: GraphNeuralNet,
    events: list,
    device: str = "cpu",
) -> float:
    """
    Compute track reconstruction accuracy across all events.

    For each true track, finds the predicted cluster with the most overlap,
    then counts how many hits match. Any missing hit (in the true track but
    not in the predicted cluster) or extra hit (in the predicted cluster but
    belonging to a different track) counts as one mistake.

    Score per track = (hits in true track) - (missing hits) - (extra hits)
    clamped to 0. Final accuracy is the sum of per-track scores divided by
    the total number of hits across all events.

    Parameters
    ----------
    model : GraphNeuralNet
        Trained model in eval mode.
    events : list of list of TrackHit
        Raw events with ground-truth track_id on each hit.
    device : str, optional (default="cpu")
        Device to run inference on.

    Returns
    -------
    float
        Fraction of hits correctly reconstructed, in [0, 1].
    """
    model.eval()
    total_score = 0
    total_hits = 0

    with torch.no_grad():
        for hits in events:
            node_features, edge_index, edge_attr = build_graph(hits)
            graph = Data(
                x=node_features, edge_index=edge_index, edge_attr=edge_attr
            ).to(device)
            logits = model(graph)

            pred_edges = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
            pred_clusters = get_predicted_clusters(
                edge_index.numpy(), pred_edges, len(hits)
            )
            true_track_ids = np.array([h.track_id for h in hits], dtype=int)

            # group node indices by true track and predicted cluster
            true_tracks = defaultdict(set)
            pred_tracks = defaultdict(set)
            for node_idx, (true_id, pred_id) in enumerate(
                zip(true_track_ids, pred_clusters)
            ):
                true_tracks[true_id].add(node_idx)
                pred_tracks[pred_id].add(node_idx)

            # for each true track, find the best matching predicted cluster
            for true_id, true_nodes in true_tracks.items():
                best_score = 0
                for pred_id, pred_nodes in pred_tracks.items():
                    correct = len(true_nodes & pred_nodes)  # hits in both
                    missing = len(true_nodes - pred_nodes)  # hits the model missed
                    extra = len(pred_nodes - true_nodes)  # hits that don't belong
                    score = correct - missing - extra
                    best_score = max(best_score, score)

                total_score += max(0, best_score)
                total_hits += len(true_nodes)

    return total_score / total_hits if total_hits > 0 else 0.0


def get_predicted_clusters(
    edge_index: torch.Tensor,
    pred_edges: torch.Tensor,
    num_nodes: int,
) -> np.ndarray:
    """
    Convert predicted edge labels into cluster assignments via connected components.

    Builds an undirected graph from the edges the model predicted as same-track
    (pred_edges == True), then finds connected components. Each component is one
    predicted track — hits that are transitively connected by high-confidence
    edges end up in the same cluster.

    This sidesteps the permutation problem: instead of predicting a fixed track
    label (which has no consistent meaning across events), the model only predicts
    pairwise same/different, and clustering recovers the groupings.

    Parameters
    ----------
    edge_index : torch.Tensor, shape (2, n_edges)
        Candidate edges as (source, destination) index pairs.
    pred_edges : torch.Tensor, shape (n_edges,), dtype bool
        Model predictions: True if the edge is predicted as same-track.
    num_nodes : int
        Total number of hits (nodes) in the event.

    Returns
    -------
    np.ndarray, shape (num_nodes,), dtype int
        Cluster ID for each node. Hits in the same connected component
        share the same ID. IDs are arbitrary integers with no physical
        meaning — only the grouping matters.
    """
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


def visual_example(
    model: GraphNeuralNet,
    graph: DataLoader,
    sample_hits: list,
    radii: np.ndarray,
    device: str,
    n_tracks: int,
) -> None:
    """
    Run the model on a single example event and save a comparison plot.

    Pulls one graph from the DataLoader, runs a forward pass, converts the
    predicted edge scores into cluster assignments via connected components,
    and calls plot_comparison to visualise ground truth alongside predictions.

    Parameters
    ----------
    model : GraphNeuralNet
        Trained model in eval mode.
    graph : DataLoader
        DataLoader wrapping a single-event HitDataset. Only the first
        batch is used.
    sample_hits : list of list of TrackHit
        The raw events corresponding to the DataLoader. The first event
        is used to extract hit positions and ground-truth track IDs.
    radii : np.ndarray
        Detector ring radii, passed through to plot_comparison.
    device : str
        Device to run inference on ("cpu" or "cuda").
    n_tracks : int
        Number of tracks in the event, used to label the output file as
        <model.name>_<n_tracks>_visual_sample.png.
    """
    model.eval()
    with torch.no_grad():
        graph = next(iter(graph)).to(device)
        logits = model(graph)

    probs = torch.sigmoid(logits)
    pred_edges = (probs > 0.5).cpu().numpy()
    pred_clusters = get_predicted_clusters(
        graph.edge_index.cpu(),  # pyright: ignore[reportAttributeAccessIssue]
        pred_edges,  # pyright: ignore[reportArgumentType]
        graph.x.size(0),  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    )
    true_clusters = np.array([h.track_id for h in sample_hits[0]])

    plot_comparison(
        sample_hits[0],
        true_clusters,
        pred_clusters,
        radii,
        f"{model.name}_{n_tracks}_visual_sample.png",
    )


def plot_comparison(
    hits: list,
    true_clusters: np.ndarray,
    pred_clusters: np.ndarray,
    radii: np.ndarray,
    file_name: str,
) -> None:
    """
    Plot ground-truth and predicted track clusters side by side.

    Produces a two-panel figure where each hit is coloured by its cluster
    assignment. Comparing the panels visually shows where the model groups
    hits correctly and where it merges or splits tracks.

    Parameters
    ----------
    hits : list of TrackHit
        All hits from a single event, used to extract (x, y) positions.
    true_clusters : np.ndarray, shape (n_hits,), dtype int
        Ground-truth cluster ID per hit, derived from track_id.
    pred_clusters : np.ndarray, shape (n_hits,), dtype int
        Predicted cluster ID per hit, from get_predicted_clusters.
    radii : np.ndarray
        Detector ring radii, used to draw the concentric circle overlays.
    file_name : str
        Output filename (saved to results/<file_name>).
    """
    pos = np.array([(h.x, h.y) for h in hits])
    fig, axes = plt.subplots(2, 1, figsize=(8, 16))

    for ax in axes:
        for r in radii:
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color="green", alpha=0.4))  # pyright: ignore[reportPrivateImportUsage]
        ax.scatter([0], [0], c="black", marker="x")
        lim = radii[-1] * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.tick_params(axis="both", labelsize=16)

    axes[0].scatter(pos[:, 0], pos[:, 1], c=true_clusters, cmap="tab10", s=40)
    axes[0].set_title("Ground truth", fontsize=20)

    axes[1].scatter(pos[:, 0], pos[:, 1], c=pred_clusters, cmap="tab10", s=40)
    axes[1].set_title("Predicted", fontsize=20)

    plt.tight_layout()
    plt.savefig(Path("results") / file_name)


def export_predictions(
    model: GraphNeuralNet,
    events: list,
    event_ids: list,
    device: str = "cpu",
    filename: str = "predictions.csv",
) -> None:
    """
    Run the model over a set of events and write predictions to a CSV file.

    Each line contains the event ID followed by hit data in the format:
        event_id, x1, y1, pred_track_id_1, x2, y2, pred_track_id_2, ...

    The predicted track ID for each hit is the connected-component cluster
    ID assigned by get_predicted_clusters — arbitrary integers with no
    physical meaning, but consistent within each event.

    Parameters
    ----------
    model : GraphNeuralNet
        Trained model in eval mode.
    events : list of list of TrackHit
        Raw events to run inference on.
    event_ids : list of int
        Corresponding event IDs, matched by index to events.
    device : str, optional (default="cpu")
        Device to run inference on.
    filename : str, optional (default="predictions.csv")
        Output filename, written to the results/ directory.
    """
    model.eval()
    model = model.to(device)

    rows = []
    with torch.no_grad():
        for event_id, hits in tqdm(
            zip(event_ids, events), total=len(events), desc="Exporting predictions"
        ):
            node_features, edge_index, edge_attr = build_graph(hits)
            graph = Data(
                x=node_features, edge_index=edge_index, edge_attr=edge_attr
            ).to(device)

            logits = model(graph)
            pred_edges = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
            clusters = get_predicted_clusters(edge_index.numpy(), pred_edges, len(hits))  # pyright: ignore[reportArgumentType]

            # build the row: event_id, x1, y1, pred_tid_1, x2, y2, pred_tid_2, ...
            row = [event_id]
            for hit, cluster_id in zip(hits, clusters):
                row += [hit.x, hit.y, cluster_id]
            rows.append(row)

    output_path = Path("results") / filename
    with open(output_path, "w") as f:
        for row in rows:
            f.write(", ".join(str(v) for v in row) + "\n")

    print(f"Predictions written to {output_path}")

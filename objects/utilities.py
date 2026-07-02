from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

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
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=0.001,
    device="cpu",
    patience=15,
    filename="GNN",
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
    with open(Path("models") / f"{filename}.txt", "w") as file:
        file.write("")
        file.close()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    model = model.to(device)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for graph in train_loader:
            graph = graph.to(device)

            optimizer.zero_grad()
            pred = model(graph)

            # Determine accuracy
            probs = torch.sigmoid(pred)
            pred_labels = (probs >= 0.5).float()
            correct = (pred_labels == graph.edge_label).sum().item()
            total = graph.edge_label.numel()
            running_correct += correct
            running_total += total

            # Back propagation
            loss = criterion(pred, graph.edge_label.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = running_correct / running_total
        model.train_loss.append(train_loss)
        model.train_acc.append(train_accuracy)

        # Validation
        model.eval()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        with torch.no_grad():
            for graph in val_loader:
                graph = graph.to(device)

                pred = model(graph)

                # Determine accuracy
                probs = torch.sigmoid(pred)
                pred_labels = (probs >= 0.5).float()
                correct = (pred_labels == graph.edge_label).sum().item()
                total = graph.edge_label.numel()
                running_correct += correct
                running_total += total

                # Determine loss
                loss = criterion(pred, graph.edge_label.float())
                running_loss += loss.item()

            val_loss = running_loss / len(val_loader)
            val_accuracy = running_correct / running_total

            model.val_loss.append(val_loss)
            model.val_acc.append(val_accuracy)

        # Validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(filename)
        elif epochs_no_improve >= patience:
            break
        else:
            epochs_no_improve += 1

        logs = f"Epoch [{epoch + 1:3d}/{epochs}] | Train loss: {train_loss:.5f} | Val loss {val_loss:.5f} | Best loss: {best_val_loss:.5f}| Train accuracy {train_accuracy:.5f} | Val accuracy {val_accuracy:.5f} | Patience: [{epochs_no_improve:2d}/{patience}]\n"
        with open(Path("models") / f"{filename}.txt", "a") as file:
            file.write(logs)
            file.close()
        print(logs, end="\r", flush=True)

    model.plot_performance(filename)


def test_model(model, test_loader, device="cpu"):
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()

    running_loss = 0
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        for graph in test_loader:
            graph = graph.to(device)

            pred = model(graph)

            loss = criterion(pred, graph.edge_label.float())
            running_loss += loss.item()

            probs = torch.sigmoid(pred)
            pred_labels = (probs >= 0.5).float()

            running_correct += (pred_labels == graph.edge_label).sum().item()
            running_total += graph.edge_label.numel()

    loss = running_loss / len(test_loader)
    acc = running_correct / running_total

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    return loss, acc

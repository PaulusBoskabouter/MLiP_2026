from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from objects.collider import CollisionDetector2D, TrackHit
from objects.network import GraphNeuralNet
from objects.utilities import HitDataset, test_model, train_model, visual_example

device = "cuda" if torch.cuda.is_available() else "cpu"


def run_experiment(
    events, model_name, detector, train, epochs, lr, patience, batch_size, seed
):
    """Shared training/eval pipeline for both parts."""
    train_data, test_data = train_test_split(
        events, test_size=0.2, random_state=seed, shuffle=True
    )
    train_data, val_data = train_test_split(
        train_data, test_size=0.125, random_state=seed, shuffle=True
    )

    if train:
        train_loader = DataLoader(
            HitDataset(train_data, desc="train"),  # pyright: ignore[reportArgumentType]
            batch_size=batch_size,
            shuffle=True,  # pyright: ignore[reportArgumentType]
        )
        val_loader = DataLoader(
            HitDataset(val_data, desc="validation"),  # pyright: ignore[reportArgumentType]
            batch_size=batch_size,
            shuffle=False,
        )
    test_loader = DataLoader(
        HitDataset(test_data, desc="test"),  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        shuffle=False,  # pyright: ignore[reportArgumentType]
    )

    print(
        f"Training {model_name}:"
        + f"\n\tepochs: {epochs}  lr: {lr}  patience: {patience}  batch: {batch_size}"
        + f"\n\ttrain/val/test: {len(train_data)}/{len(val_data)}/{len(test_data)}"
        + f"\n\tdevice: {device}\n"
    )

    model = GraphNeuralNet(name=model_name)
    if train:
        train_model(
            model,
            train_loader,  # pyright: ignore[reportPossiblyUnboundVariable]
            val_loader,  # pyright: ignore[reportPossiblyUnboundVariable]
            epochs=epochs,
            lr=lr,
            device=device,
            patience=patience,
        )
    else:
        model.load()

    loss, f1 = test_model(model, test_loader, device=device)
    print(f"Test — loss: {loss:.4f}  F1: {f1:.4f}\n")

    sample = HitDataset(events[:1])
    visual_example(model, sample, events[:1], detector.radii, device, detector.n_tracks)
    print(
        f"Visual example saved to {Path('results') / f'{model.name}_visual_sample.png'}"
    )
    print("=" * 100)
    print()


def model_linear_tracks(seed, train, epochs, lr, patience, batch_size, n_tracks):
    detector = CollisionDetector2D(
        n_layers=5, r_min=1.0, r_max=6.0, n_tracks=n_tracks, random_seed=seed
    )
    detector.plot_event(
        detector.generate_event(),
        filename=f"example_{n_tracks}_tracks",
    )

    data = detector.generate_dataset(n_events=100)
    detector.histogram_dataset(data, f"distribution_{n_tracks}_tracks")
    print(f"Generated {len(data)} events with {n_tracks} tracks.")
    print("-" * 100)

    run_experiment(
        data,
        "GNN_linear_tracks",
        detector,
        train,
        epochs,
        lr,
        patience,
        batch_size,
        seed,
    )


def model_curved_tracks(seed, train, epochs, lr, patience, batch_size):
    print("Loading curved track data from 'hits_FINAL.csv'...")
    df = pd.read_csv("hits_FINAL.csv")

    events = []
    for event_id in df["event_id"].unique():
        event_df = df[df["event_id"] == event_id]
        hits = [
            TrackHit(node_id, row["x"], row["y"], row["track_id"], row["layer"])
            for node_id, (_, row) in enumerate(event_df.iterrows())
        ]
        events.append(hits)
        if len(events) > 100:
            break
    detector = CollisionDetector2D()
    detector.plot_event(events[seed], filename="example_curved_tracks")
    print("-" * 100)

    run_experiment(
        events,
        "GNN_curved_tracks",
        detector,
        train,
        epochs,
        lr,
        patience,
        batch_size,
        seed,
    )


if __name__ == "__main__":
    # HYPERPARAMETERS:
    seed = 42
    train = True
    epochs = 1000
    lr = 0.001
    patience = 15
    batch_size = 32

    # Training and evaluating the model with 10 tracks
    model_linear_tracks(
        seed=seed,
        train=train,
        epochs=epochs,
        lr=lr,
        patience=patience,
        batch_size=batch_size,
        n_tracks=10,
    )
    # Evaluating the model with 50 tracks
    model_linear_tracks(
        seed=seed,
        train=False,
        epochs=epochs,
        lr=lr,
        patience=patience,
        batch_size=batch_size,
        n_tracks=50,
    )
    # Training and evaluating the model with curved tracks
    model_curved_tracks(
        seed=seed,
        train=train,
        epochs=epochs,
        lr=lr,
        patience=patience,
        batch_size=batch_size,
    )

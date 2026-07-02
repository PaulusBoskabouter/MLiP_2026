# base imports
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from objects.collider import CollisionDetector, TrackHit
from objects.network import GraphNeuralNet
from objects.utilities import HitDataset, test_model, train_model, visual_example

device = "cuda" if torch.cuda.is_available() else "cpu"


def model_linear_tracks(seed, train, epochs, lr, patience, n_tracks):
    detector = CollisionDetector(
        n_layers=5, r_min=1.0, r_max=6.0, n_tracks=n_tracks, random_seed=seed
    )

    hits = detector.generate_event()
    detector.plot_event(
        hits, filename=f"example_{n_tracks}_tracks", color_by_track=True
    )
    print(
        "generating dataset of 10.000 events with the following parameters:"
        + f"\n\tlayers:\t{detector.n_layers}"
        + f"\n\ttracks:\t{detector.n_tracks}"
        + f"\n\tseed:\t{detector.seed}\n"
    )
    data = detector.generate_dataset(n_events=10000)
    detector.histogram_dataset(data, f"distribution_{n_tracks}_tracks")
    print(
        f"hit x,y and phi distributions can be found at {Path('results') / f'distribution_{n_tracks}_tracks.png'}"
    )
    print(
        f"example event can be found at: {Path('results') / f'example_{n_tracks}_tracks.png'}"
    )
    print("-" * 100)

    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=seed, shuffle=True
    )

    train_data, val_data = train_test_split(
        train_data,
        test_size=0.125,  # 0.125 * 0.8 = 0.1
        random_state=seed,
        shuffle=True,
    )

    train_loader = HitDataset(train_data)
    val_loader = HitDataset(val_data)
    test_loader = HitDataset(test_data)
    print(
        "Training the model with following parameters:"
        + f"\n\tepochs:\t{patience}"
        + f"\n\tlearning rate:\t{patience}"
        + f"\n\tpatience:\t{patience}"
        + f"\n\ttrain sample count:\t{len(train_loader)}"
        + f"\n\tval sample count:\t{len(val_loader)}"
        + f"\n\ttest sample count:\t{len(test_loader)}"
        + f"\n\tdevice:\t{device}\n"
    )
    model = GraphNeuralNet(name=f"GNN_part_1_{n_tracks}_tracks")
    if train:
        train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            patience=15,
        )
    else:
        model.load()
    print("-" * 100)
    loss, f1 = test_model(model, test_loader, device=device)
    print(
        "Testing model on test set:"
        + f"\n\tLoss:\t{loss:4f}"
        + f"\n\tF1 score:\t{f1:.4f}\n"
    )

    sample_hits = detector.generate_dataset(n_events=1)
    sample = HitDataset(sample_hits)

    visual_example(model, sample, sample_hits, detector.radii, device)
    print(
        f"Created a visual example under {Path('results') / f'{model.name}_visual_sample.png'}"
    )
    print("=" * 100)


def model_curved_tracks(seed, train, epochs, lr, patience):
    print("Loading data from 'hits_FINAL.csv'")

    df = pd.read_csv("hits_FINAL.csv")

    events_ids = df["event_id"].unique()

    events = []

    node_id = 0

    for event in events_ids:
        hits = []
        event_hits = df.loc[df["event_id"] == event]
        for __, hit in event_hits.iterrows():
            hits.append(
                TrackHit(node_id, hit["x"], hit["y"], hit["track_id"], hit["layer"])
            )
            node_id += 1

        events.append(hits)

    detector = (
        CollisionDetector()
    )  # Parameters are left standard as they are not used anyway.

    detector.plot_event(events[seed], filename="example_curved_tracks")
    print(
        f"Done! Example event can be found under f{Path('results') / 'example_curved.png'}"
    )
    print("-" * 100)
    train_data, test_data = train_test_split(
        events, test_size=0.2, random_state=seed, shuffle=True
    )

    train_data, val_data = train_test_split(
        train_data,
        test_size=0.125,  # 0.125 * 0.8 = 0.1
        random_state=seed,
        shuffle=True,
    )

    train_loader = HitDataset(train_data)
    val_loader = HitDataset(val_data)
    test_loader = HitDataset(test_data)

    print(
        "Training the model with following parameters:"
        + f"\n\tepochs:\t{patience}"
        + f"\n\tlearning rate:\t{patience}"
        + f"\n\tpatience:\t{patience}"
        + f"\n\ttrain sample count:\t{len(train_loader)}"
        + f"\n\tval sample count:\t{len(val_loader)}"
        + f"\n\ttest sample count:\t{len(test_loader)}"
        + f"\n\tdevice:\t{device}\n"
    )

    model = GraphNeuralNet(name="GNN_part_2")

    if train:
        train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            patience=15,
        )
    else:
        model.load()
    print("-" * 100)
    loss, f1 = test_model(model, test_loader, device=device)
    print(
        "Testing model on test set:"
        + f"\n\tLoss:\t{loss:4f}"
        + f"\n\tF1 score:\t{f1:.4f}\n"
    )

    sample_hits = events[:1]
    sample = HitDataset(sample_hits)

    visual_example(model, sample, sample_hits, detector.radii, device)
    print(
        f"Created a visual example under {Path('results') / f'{model.name}_visual_sample.png'}"
    )
    print("=" * 100)


if __name__ == "__main__":
    # HYPERPARAMETERS:
    seed = 42
    train = True
    epochs = 1000
    lr = 0.001
    patience = 15

    # Training and evaluating the model with 10 tracks
    model_linear_tracks(seed, train, epochs, lr, patience, n_tracks=10)
    # Training and evaluating the model with 50 tracks
    model_linear_tracks(seed, train, epochs, lr, patience, n_tracks=50)
    model_curved_tracks(seed, train, epochs, lr, patience)

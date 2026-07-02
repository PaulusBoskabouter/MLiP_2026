# base imports
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from objects.collider import CollisionDetector
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
    print("-" * 80)

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
            epochs=2,
            lr=0.001,
            device=device,
            patience=15,
        )
    else:
        model.load()
    print("-" * 80)
    loss, f1 = test_model(model, test_loader, device=device)
    print(
        "Testing model on test set:"
        + f"\n\tLoss:\t{loss:4f}"
        + f"\n\tF1 score:\t{f1:.4f}\n"
    )

    sample_hits = detector.generate_dataset(n_events=1)
    sample = HitDataset(sample_hits)

    visual_example(model, sample, sample_hits, detector.radii)
    print(
        f"Created a visual example under {Path('results') / f'{model.name}_visual_sample.png'}"
    )
    print("=" * 80)


if __name__ == "__main__":
    # HYPERPARAMETERS:
    seed = 42
    train = True
    epochs = 1000
    lr = 0.001
    device = device
    patience = 15

    model_linear_tracks(seed, train, epochs, lr, patience, n_tracks=10)
    model_linear_tracks(seed, train, epochs, lr, patience, n_tracks=50)

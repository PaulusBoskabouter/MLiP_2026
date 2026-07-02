# Base imports
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from objects.collider import CollisionDetector
from objects.network import GraphNeuralNet
from objects.utilities import HitDataset, test_model, train_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(seed):
    detector = CollisionDetector(
        n_layers=5, r_min=1.0, r_max=6.0, n_tracks=10, random_seed=seed
    )

    hits = detector.generate_event()
    detector.plot_event(hits, color_by_track=True)

    print("=" * 80)
    print(
        "Generating first dataset of 10.000 events with the following parameters:"
        + f"\nlayers:\t{detector.n_layers}"
        + f"\ntracks:\t{detector.n_tracks}"
        + f"\nseed:\t{detector.seed}\n"
    )
    data = detector.generate_dataset(n_events=100)
    detector.histogram_dataset(data)
    print(
        f"Hit x,y and phi distributions can be found at {Path('results') / 'distribution.png'}"
    )
    print(f"Example event can be found at: {Path('results') / 'example.png'}")
    print("=" * 80)

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

    model = GraphNeuralNet()
    train_model(
        model,
        train_loader,
        val_loader,
        epochs=10,
        lr=0.001,
        device=device,
        patience=15,
        filename="GNN_part_1_10_tracks",
    )

    test_model(model, test_loader, device=device)

    visual_sample = ...


if __name__ == "__main__":
    seed = 42
    main(seed)

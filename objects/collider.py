from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from objects.graph import TrackHit


class CollisionDetector:
    """
    Simplified Monte Carlo simulator for particle tracks and detector hits.
    """

    def __init__(self, n_layers=5, r_min=1.0, r_max=10.0, n_tracks=10, random_seed=42):
        """
        Parameters
        ----------
        n_layers : int
            Number of concentric detector circles.
        r_min, r_max : float
            Radius of the innermost / outermost detector circle.
        n_tracks : int
            Number of tracks to generate per event (fixed at 3 for part a).
        random_seed : int or None
            Seed for reproducibility.
        """
        self.n_layers = n_layers
        self.radii = np.linspace(r_min, r_max, n_layers)
        self.n_tracks = n_tracks
        self.seed = random_seed
        self.rng = np.random.default_rng(self.seed)

    def generate_tracks(self):
        """
        Generates `self.n_tracks` random track angles phi in [0, 2*pi).

        Returns
        -------
        phis : np.ndarray, shape (n_tracks,)
        """

        return self.rng.uniform(low=0.0, high=2 * np.pi, size=(self.n_tracks,))

    def generate_event(self):
        """
        Function that:
          1. sample track angles via generate_tracks()
          2. for each track, compute its hits via
          3. collect hits into a single array (with 95% chance)
          4. also keep a track_id label per hit (ground truth, for training/eval only)

        Returns
        -------
        hits : np.ndarray, shape (n_tracks * n_layers, 2)
            All (x, y) hit coordinates, order shuffled (a real detector
            doesn't hand them to you pre-sorted by track!).
        labels : np.ndarray, shape (n_tracks * n_layers,)
            track_id for each hit, in the same shuffled order as `hits`.
        """

        all_hits = []
        tracks = self.generate_tracks()

        node_id = 0
        for id, track_phi in enumerate(tracks):
            for ring_id, r in enumerate(self.radii):  # generate the hits
                # With 95% chance of recording the hit:
                if self.rng.uniform(0.0, 1.0) <= 0.95:
                    # Determine position
                    x = r * np.cos(track_phi)
                    y = r * np.sin(track_phi)

                    # Add the noise to the measurments
                    sigma = r * 0.1e-2  # 0.1% of the radius
                    x += self.rng.normal(scale=sigma)
                    y += self.rng.normal(scale=sigma)
                    hit = TrackHit(node_id, x, y, id, ring_id)
                    all_hits.append(hit)
                    node_id += 1

        return all_hits

    def plot_event(self, hits, filename, color_by_track=True):
        """
        Quick visualization: detector circles + hits, optionally colored by track_id.
        """
        fig, ax = plt.subplots(figsize=(6, 6))

        # detector circles
        for r in self.radii:
            circle = plt.Circle((0, 0), r, fill=False, color="green", alpha=0.4)  # pyright: ignore[reportPrivateImportUsage]
            ax.add_patch(circle)

        # hits
        pos = np.array([(hit.x, hit.y) for hit in hits])
        if color_by_track:
            idx = np.array([hit.track_id for hit in hits])
            ax.scatter(
                pos[:, 0],
                pos[:, 1],
                c=idx,
                cmap="tab10",
                s=40,
                zorder=3,
            )
        else:
            ax.scatter(pos[:, 0], pos[:, 1], c="orange", s=40, zorder=3)

        ax.scatter([0], [0], c="black", marker="x", s=60, label="origin")
        ax.set_xlim(-self.radii[-1] * 1.2, self.radii[-1] * 1.2)
        ax.set_ylim(-self.radii[-1] * 1.2, self.radii[-1] * 1.2)
        ax.set_aspect("equal")
        ax.legend()
        plt.savefig(Path("results") / f"{filename}.png")
        return fig, ax

    def histogram_dataset(self, dataset, filename):
        all_hits = []
        all_phis = []
        for event in dataset:  # dataset is your list of TrackHit lists
            for hit in event:
                all_hits.append([hit.x, hit.y])
                all_phis.append(hit.phi)

        all_hits = np.array(all_hits)  # shape (N_total_hits, 2)
        all_phis = np.array(all_phis)  # shape (N_total_hits, 2)
        x = all_hits[:, 0]
        y = all_hits[:, 1]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].hist(x, bins=100)
        axes[0].set_title("Hit x distribution")
        axes[1].hist(y, bins=100)
        axes[1].set_title("Hit y distribution")
        axes[2].hist(all_phis, bins=100)
        axes[2].set_title("Hit phi distribution")
        plt.tight_layout()
        plt.savefig(Path("results") / f"{filename}.png")

    def generate_dataset(self, n_events=10000):
        events = []
        for _ in range(n_events):
            hits = self.generate_event()
            events.append(hits)
        return events

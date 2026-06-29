from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class CollisionDetector:
    """
    Simplified Monte Carlo simulator for particle tracks and detector hits.

    Setup (part a):
      - No magnetic field -> tracks are straight lines (rays) from the origin.
      - Detector = `n_layers` concentric circles around the origin.
      - A "hit" is the point where a track ray crosses a detector circle.
    """

    def __init__(self, n_layers=5, r_min=1.0, r_max=5.0, n_tracks=3, random_seed=None):
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
        self.rng = np.random.default_rng(random_seed)

    def generate_tracks(self):
        """
        TODO: generate `self.n_tracks` random track angles theta in [0, 2*pi).

        Returns
        -------
        thetas : np.ndarray, shape (n_tracks,)
        """

        return np.random.uniform(low=0.0, high=2 * np.pi, size=(self.n_tracks,))

    def track_to_hits(self, theta):
        """
        TODO: given a track angle theta, compute the (x, y) hit position
        on each of the `self.radii` circles.

        A point on a circle of radius r at angle theta is:
            x = r * cos(theta)
            y = r * sin(theta)

        Returns
        -------
        hits : np.ndarray, shape (n_layers, 2)
        """
        l = []
        for r in self.radii:
            l.append([r * np.cos(theta), r * np.sin(theta)])

        return np.array(l)

    def generate_event(self):
        """
        TODO: generate one full event:
          1. sample track angles via generate_tracks()
          2. for each track, compute its hits via track_to_hits()
          3. collect ALL hits into a single array (the "detector view" -
             i.e. what you'd actually observe, unlabeled)
          4. also keep a track_id label per hit (ground truth, for training/eval only)

        Returns
        -------
        hits : np.ndarray, shape (n_tracks * n_layers, 2)
            All (x, y) hit coordinates, order shuffled (a real detector
            doesn't hand them to you pre-sorted by track!).
        labels : np.ndarray, shape (n_tracks * n_layers,)
            track_id for each hit, in the same shuffled order as `hits`.
            This is your ground truth, not detector input.
        """

        hits = []
        labels = []
        tracks = self.generate_tracks()

        for id, track in enumerate(tracks):
            hits.extend(self.track_to_hits(track))
            labels.extend([id for i in range(self.n_layers)])

        return np.array(hits), np.array(labels)

    def plot_event(self, hits, labels=None):
        """
        Quick visualization: detector circles + hits, optionally colored by track_id.
        """
        fig, ax = plt.subplots(figsize=(6, 6))

        # detector circles
        for r in self.radii:
            circle = plt.Circle((0, 0), r, fill=False, color="green", alpha=0.4)
            ax.add_patch(circle)

        # hits
        if labels is not None:
            ax.scatter(hits[:, 0], hits[:, 1], c=labels, cmap="tab10", s=40, zorder=3)
        else:
            ax.scatter(hits[:, 0], hits[:, 1], c="orange", s=40, zorder=3)

        ax.scatter([0], [0], c="black", marker="x", s=60, label="origin")
        ax.set_xlim(-self.radii[-1] * 1.2, self.radii[-1] * 1.2)
        ax.set_ylim(-self.radii[-1] * 1.2, self.radii[-1] * 1.2)
        ax.set_aspect("equal")
        ax.legend()
        plt.savefig(Path("results") / "test.png")
        return fig, ax


if __name__ == "__main__":
    det = CollisionDetector(
        n_layers=5, r_min=1.0, r_max=5.0, n_tracks=3, random_seed=42
    )
    hits, labels = det.generate_event()
    print("hits shape:", hits.shape)
    print("labels:", labels)
    det.plot_event(hits, labels)

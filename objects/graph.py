from collections import defaultdict

import numpy as np
import torch


class TrackHit:
    """
    Simple object for keeping neatly track of all individual hits and the track they belong to. This object also serves as a node for the graph.
    """

    def __init__(self, node_id: int, x: float, y: float, track_id: int, ring: int):
        """
        Parameters
        ----------
        node_id : int
            Identifier to know which index of the all_hits list belongs to this node.
        x : float
            x coordinate of hit location.
        y : float
            y coordinate of hit location.
        track_id : int
            the integer label for determining connections (and thus labels).
        ring : int
            Identifier of the which ring this hit was measured on, this is used for determining node-to-node distances.
        """
        self.id = node_id

        self.x = x
        self.y = y
        self.track_id = track_id
        # Reason we recalculate phi instead of taking the label directly
        # is of course because our measurements (x and y) contain some noise.
        # Therefore, not re-calculating would be cheating a bit.
        self.phi = np.arctan2(y, x)
        self.ring = ring


def build_graph(
    event: list, k: int = 3, max_ring_skip: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a list of TrackHit objects into a graph for the GNN.

    Each hit becomes a node with features [x_norm, y_norm]. Candidate edges
    are formed by connecting each hit to its k nearest neighbours (by Euclidean
    distance) on rings within max_ring_skip in each direction. All hits on
    those target rings are considered as candidates and the k closest win. This ring-skipping ensures that if a
    hit is missing (due to 5% dropout), tracks don't become disconnected: a hit on
    ring 0 can still reach ring 2 if ring 1 was empty.

    Edge labels are assigned purely by track membership: 1 if both endpoints
    share the same track_id, 0 otherwise. Labels are only used during
    training; at inference the GNN predicts them.

    Parameters
    ----------
    event : list of TrackHit
        All hits from a single collision event.
    k : int, optional (default=3)
        Number of nearest neighbours to connect per hit per direction.
        Higher k gives the GNN more candidate edges to score but increases
        the fraction of false (different-track) edges.
    max_ring_skip : int, optional (default=2)
        How many rings away to look for candidates in each direction.
        1 = adjacent rings only; 2 = also look two rings away as a fallback
        for missed hits. Handles up to (max_ring_skip - 1) consecutive
        missed hits per track.

    Returns
    -------
    node_features : torch.Tensor, shape (n_hits, 2)
        Per-node feature vectors [x / r_max, y / r_max].
    edge_index : torch.Tensor, shape (2, n_edges), dtype long
        Candidate edges as (source, destination) index pairs. Both directions
        are included so every node can receive messages from its neighbours.
    edge_labels : torch.Tensor, shape (n_edges,), dtype float32
        Ground-truth edge labels: 1.0 if the two endpoints belong to the
        same track, 0.0 otherwise.
    """

    xs = np.array([h.x for h in event], dtype=np.float32)
    ys = np.array([h.y for h in event], dtype=np.float32)
    rings = np.array([h.ring for h in event], dtype=np.int32)
    tids = np.array([h.track_id for h in event])

    r_max = np.sqrt(xs**2 + ys**2).max()
    r_max = r_max if r_max > 0 else 1.0

    node_features = np.stack([xs / r_max, ys / r_max], axis=1).astype(np.float32)

    # Group hit indices by ring for fast lookup
    ring_to_idxs = defaultdict(list)
    for idx, ring in enumerate(rings):
        ring_to_idxs[int(ring)].append(idx)

    edges = set()

    for i, h in enumerate(event):
        for direction in (+1, -1):
            cand_idxs = []
            cand_dist = []

            for skip in range(1, max_ring_skip + 1):
                target_ring = h.ring + direction * skip
                for j in ring_to_idxs.get(target_ring, []):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    cand_idxs.append(j)
                    cand_dist.append(
                        dx * dx + dy * dy
                    )  # squared distance, no sqrt needed for ranking

            if not cand_idxs:
                continue

            # k nearest by Euclidean distance across the full candidate pool
            for ni in np.argsort(cand_dist)[:k]:
                j = cand_idxs[ni]
                edges.add((i, j))
                edges.add((j, i))

    edge_index = np.array(list(edges), dtype=np.int64).T
    edge_labels = (tids[edge_index[0]] == tids[edge_index[1]]).astype(np.float32)

    return (
        torch.tensor(node_features, dtype=torch.float32),
        torch.tensor(edge_index, dtype=torch.long),
        torch.tensor(edge_labels, dtype=torch.float32),
    )

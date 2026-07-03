from collections import defaultdict

import numpy as np
import torch


class TrackHit:
    """
    Simple object for keeping neatly track of all individual hits and the track they belong to. This object also serves as a node for the graph.
    """

    def __init__(self, node_id, x, y, track_id, ring):
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


def build_graph(event: list, k: int = 3, max_ring_skip: int = 2):
    """
    Parameters
    ----------
    k              : k-nearest neighbours by phi per direction per source hit
    max_ring_skip  : look up to this many rings away in each direction.
                        1 = only adjacent rings; 2 = handles single missed hits.
    """
    if not event:
        return torch.zeros(0, 5), torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

    # --- node feature arrays ---
    xs = np.array([h.x for h in event])
    ys = np.array([h.y for h in event])
    phis = np.array([h.phi for h in event])
    rings = np.array([h.ring for h in event])
    tids = np.array([h.track_id for h in event])

    r = np.sqrt(xs**2 + ys**2)
    r_max = r.max() if r.max() > 0 else 1.0

    node_features = np.stack(
        [
            xs / r_max,  # x normalised
            ys / r_max,  # y normalised
        ],
        axis=1,
    ).astype(np.float32)

    # --- index map and ring lookup ---
    ring_to_idxs = defaultdict(list)
    for idx, r_val in enumerate(rings):
        ring_to_idxs[int(r_val)].append(idx)

    # --- edge construction ---
    edges = set()

    for i, h in enumerate(event):
        for direction in (+1, -1):
            # Collect ALL candidate indices from rings within max_ring_skip
            cand_idxs = []
            cand_dphi = []

            for skip in range(1, max_ring_skip + 1):
                target_ring = h.ring + direction * skip
                for j in ring_to_idxs.get(target_ring, []):
                    d = abs(phis[i] - phis[j])
                    d = min(d, 2 * np.pi - d)
                    cand_idxs.append(j)
                    cand_dphi.append(d)

            if not cand_idxs:
                continue

            # k-nearest by phi across the whole candidate pool
            order = np.argsort(cand_dphi)[:k]
            for ni in order:
                j = cand_idxs[ni]
                edges.add((i, j))
                edges.add((j, i))  # undirected → both directions

    if not edges:
        return (
            torch.tensor(node_features),
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0),
        )

    edge_index = np.array(list(edges), dtype=np.int64).T
    edge_labels = (tids[edge_index[0]] == tids[edge_index[1]]).astype(np.float32)

    return (
        torch.tensor(node_features, dtype=torch.float32),
        torch.tensor(edge_index, dtype=torch.long),
        torch.tensor(edge_labels, dtype=torch.float32),
    )

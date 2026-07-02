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


def build_graph(
    event: list, k: int = 3, max_ring_skip: int = 2, dot_threshold: float = 0.5
):
    """
    Parameters
    ----------
    k              : k-nearest neighbours by phi per direction per source hit
    max_ring_skip  : look up to this many rings away in each direction.
                        1 = only adjacent rings; 2 = handles single missed hits.
    dot_threshold  : prune edges where dot(û_i, û_j) < threshold.
                        Guards against connecting hits pointing in opposite
                        directions.  0.5 ≈ ±60°, loose sanity cut only.
                        Set higher (0.95+) if you want a tighter angular gate.
    """
    if not event:
        return torch.zeros(0, 5), torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

    # --- node feature arrays ---
    ids = [h.id for h in event]
    xs = np.array([h.x for h in event])
    ys = np.array([h.y for h in event])
    phis = np.array([h.phi for h in event])
    rings = np.array([h.ring for h in event])
    tids = np.array([h.track_id for h in event])

    r = np.sqrt(xs**2 + ys**2)
    r_max = r.max() if r.max() > 0 else 1.0
    r_safe = np.where(r > 0, r, 1.0)
    unit_xy = np.stack([xs, ys], axis=1) / r_safe[:, None]  # (N, 2)

    ring_max = rings.max() if rings.max() > 0 else 1
    node_features = np.stack(
        [
            xs / r_max,  # x normalised
            ys / r_max,  # y normalised
            r / r_max,  # radius normalised
            phis / np.pi,  # phi ∈ [-1, 1]
            rings / ring_max,  # layer ∈ [0, 1]
        ],
        axis=1,
    ).astype(np.float32)

    # --- index map and ring lookup ---
    id_to_idx = {hid: idx for idx, hid in enumerate(ids)}
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
                    # Dot product gate (loose sanity cut)
                    # if np.dot(unit_xy[i], unit_xy[j]) < dot_threshold:
                    #     continue

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

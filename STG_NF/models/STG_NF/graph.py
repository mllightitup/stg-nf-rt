# Graph definitions, based on awesome previous work by https://github.com/yysijie/st-gcn
import numpy as np


class Graph:
    """The Graph to models the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes

    """

    def __init__(
        self, layout="openpose", strategy="spatial", headless=False, max_hop=1
    ):
        self.headless = headless
        self.max_hop = max_hop
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return str(self.A)

    def get_edge(self, layout):
        if layout == "alphapose":
            self.num_node = 17
            neighbor_link = [
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 4),
                (5, 6),
                (5, 7),
                (7, 9),
                (6, 8),
                (8, 10),
                (11, 13),
                (12, 14),
                (13, 15),
                (14, 16),
            ]
            if self.headless:
                neighbor_link = [
                    (0, 1),
                    (0, 2),
                    (2, 4),
                    (1, 3),
                    (3, 5),
                    (6, 8),
                    (7, 9),
                    (8, 10),
                    (9, 11),
                ]
                self.num_node = 14
            self_link = [(i, i) for i in range(self.num_node)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == "openpose":
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (4, 3),
                (3, 2),
                (7, 6),
                (6, 5),
                (13, 12),
                (12, 11),
                (10, 9),
                (9, 8),
                (11, 5),
                (8, 2),
                (5, 1),
                (2, 1),
                (0, 1),
                (15, 0),
                (14, 0),
                (17, 15),
                (16, 14),
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == "ntu-rgb+d":
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                (1, 2),
                (2, 21),
                (3, 21),
                (4, 3),
                (5, 21),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 21),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 1),
                (14, 13),
                (15, 14),
                (16, 15),
                (17, 1),
                (18, 17),
                (19, 18),
                (20, 19),
                (22, 23),
                (23, 8),
                (24, 25),
                (25, 12),
            ]
            # Convert 1-based to 0-based indexing.
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        else:
            raise ValueError("Layout not recognized.")

    def get_adjacency(self, strategy):
        # Build basic adjacency: edge exists if hop distance <= max_hop.
        adjacency = (self.hop_dis <= self.max_hop).astype(np.float32)
        norm_adj = normalize_digraph(adjacency)

        valid_hop = np.arange(0, self.max_hop + 1)
        if strategy == "uniform":
            self.A = norm_adj[None, ...]
        elif strategy == "distance":
            A = np.zeros(
                (len(valid_hop), self.num_node, self.num_node), dtype=np.float32
            )
            for idx, hop in enumerate(valid_hop):
                mask = self.hop_dis == hop
                A[idx][mask] = norm_adj[mask]
            self.A = A
        elif strategy == "spatial":
            center_dist = self.hop_dis[:, self.center]
            A_list = []
            for hop in valid_hop:
                mask = self.hop_dis == hop
                # Vectorized masks for root, close, and further.
                mask_root = mask & (center_dist[:, None] == center_dist[None, :])
                mask_close = mask & (center_dist[:, None] > center_dist[None, :])
                mask_further = mask & (center_dist[:, None] < center_dist[None, :])
                a_root = np.where(mask_root, norm_adj, 0)
                a_close = np.where(mask_close, norm_adj, 0)
                a_further = np.where(mask_further, norm_adj, 0)
                if hop == 0:
                    A_list.append(a_root)
                else:
                    A_list.append(a_root + a_close)
                    A_list.append(a_further)
            self.A = np.stack(A_list)
        else:
            raise ValueError("Strategy not recognized.")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1

    hop_dis = np.full((num_node, num_node), np.inf)
    # Precompute matrix powers up to max_hop.
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, axis=0)
    # Avoid division by zero.
    with np.errstate(divide="ignore"):
        inv_Dl = np.where(Dl > 0, 1.0 / Dl, 0)
    Dn = np.diag(inv_Dl)
    return np.dot(A, Dn)

import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torchph.pershom import vr_persistence, vr_persistence_l1

class UnionFind:
    '''
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    '''

    def __init__(self, n_vertices):
        '''
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        '''

        self._parent = np.arange(n_vertices, dtype=int)

    def find(self, u):
        '''
        Finds and returns the parent of u with respect to the hierarchy.
        '''

        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        '''
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        '''

        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        '''
        Generator expression for returning roots, i.e. components that
        are their own parents.
        '''

        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex


class PersistentHomologyCalculation:
    def __call__(self, matrix):

        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices)

        triu_indices = np.triu_indices_from(matrix)
        edge_weights = matrix[triu_indices]
        edge_indices = np.argsort(edge_weights, kind='stable')

        # 1st dimension: 'source' vertex index of edge
        # 2nd dimension: 'target' vertex index of edge
        persistence_pairs = []

        for edge_index, edge_weight in \
                zip(edge_indices, edge_weights[edge_indices]):

            u = triu_indices[0][edge_index]
            v = triu_indices[1][edge_index]

            younger_component = uf.find(u)
            older_component = uf.find(v)

            # Not an edge of the MST, so skip it
            if younger_component == older_component:
                continue
            elif younger_component > older_component:
                uf.merge(v, u)
            else:
                uf.merge(u, v)

            if u < v:
                persistence_pairs.append((u, v))
            else:
                persistence_pairs.append((v, u))

        # Return empty cycles component
        return np.array(persistence_pairs), np.array([])

class PH_Reg(nn.Module):
    """Topological signature."""

    def __init__(self, sort_selected=False, use_cycles=False,
                 match_edges=None):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.use_cycles = use_cycles

        self.match_edges = match_edges

        # if use_cycles:
        #     use_aleph = True
        # else:
        #     if not sort_selected and match_edges is None:
        #         use_aleph = True
        #     else:
        #         use_aleph = False

        # if use_aleph:
        #     print('Using aleph to compute signatures')
        ##self.signature_calculator = AlephPersistenHomologyCalculation(
        ##    compute_cycles=use_cycles, sort_selected=sort_selected)
        # else:
        print('Using python to compute signatures')
        self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances):
        pairs_0, pairs_1 = self.signature_calculator(
            distances.detach().cpu().numpy())

        return pairs_0, pairs_1

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        # Split 0th order and 1st order features (edges and cycles)
        pairs_0, pairs_1 = pairs
        selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]

        if self.use_cycles:
            edges_1 = distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
            edges_2 = distance_matrix[(pairs_1[:, 2], pairs_1[:, 3])]
            edge_differences = edges_2 - edges_1

            selected_distances = torch.cat(
                (selected_distances, edge_differences))

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        # return ((signature1 - signature2)**2).sum(dim=-1)
        return ((signature1 - signature2)**2).mean(dim=-1)

    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        def to_set(array):
            return set(tuple(elements) for elements in array)
        return float(len(to_set(pairs1).intersection(to_set(pairs2))))

    @staticmethod
    def _get_nonzero_cycles(pairs):
        all_indices_equal = np.sum(pairs[:, [0]] == pairs[:, 1:], axis=-1) == 3
        return np.sum(np.logical_not(all_indices_equal))

    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x = x/torch.max(x)
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances

    @staticmethod
    def sample_features(W, nSamples):
        n = W.shape[0]
        random_indices = np.random.choice(n, size=nSamples, replace=False)
        return random_indices

    def forward(self, features, gt, d_size=8,min_points=200, max_points=1000,
                         point_jump=50, train_labels=None):

        if max_points > features.shape[0]:
            max_points = features.shape[0]

        # features = features / torch.max(features)
        # label = label / torch.max(label)

        tmp = torch.pow(features,2)
        tmp = torch.sum(tmp, dim=1)
        tmp = torch.sqrt(tmp)
        features = features/ torch.max(tmp)

        gt = gt/ torch.max(gt)

        """
        Randomly sampling for PH-Dimension
        """
        test_n = range(min_points, max_points, point_jump)
        lengths = []

        for n in test_n:
            random_indices = self.sample_features(features, n)
            samples_feautre = features[random_indices]
            dist_matrix = torch.cdist(samples_feautre, samples_feautre)
            d, _ = vr_persistence(dist_matrix, 0, 0)
            d = d[0]
            d = (d[:, 1] - d[:, 0]).sum()

            samples_gt = gt[random_indices]
            dist_matrix = torch.cdist(samples_gt, samples_gt)
            d2, _ = vr_persistence(dist_matrix, 0, 0)
            d2 = d2[0]
            d2 = (d2[:, 1] - d2[:, 0]).sum()
            lengths.append(d / d2)

        lengths = torch.stack(lengths)

        # compute ph dim by running a linear least squares
        x = torch.tensor(test_n).to(lengths).log()
        y = lengths.log()
        N = len(x)
        phd_m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
        phd_m = torch.abs(phd_m)

        """
        # Topology Autoencoder 
        # """
        # samples = random.sample(range(0, features.size()[0] - 1), max_points)
        # gt = gt[samples, :]
        # features = features[samples, :]

        distances1 = self._compute_distance_matrix(features)
        distances2 = self._compute_distance_matrix(gt)

        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)
        sig1 = self._select_distances_from_pairs(distances1, pairs1)
        sig2 = self._select_distances_from_pairs(distances2, pairs2)

        sig1_2 = self._select_distances_from_pairs(distances1, pairs2)
        sig2_1 = self._select_distances_from_pairs(distances2, pairs1)

        m = self.sig_error(sig2, sig1_2) + self.sig_error(sig1, sig2_1)
        return [m, phd_m]

import numpy as np
import torch
import torch.nn as nn
import random
from torchph.pershom import vr_persistence

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
    def __init__(self):
        super().__init__()
        self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances):
        pairs_0, pairs_1 = self.signature_calculator(
            distances.detach().cpu().numpy())

        return pairs_0, pairs_1

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        # Split 0th order and 1st order features (edges and cycles)
        pairs_0, pairs_1 = pairs
        selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        # return ((signature1 - signature2)**2).sum(dim=-1)
        return ((signature1 - signature2)**2).mean(dim=-1)

    @staticmethod
    def _compute_distance_matrix(x, p=2):

        tmp = torch.pow(x,2)
        tmp = torch.sum(tmp, dim=1)
        tmp = torch.sqrt(tmp)
        x = x/ torch.max(tmp)

        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances

    @staticmethod
    def sample_features(W, nSamples):
        n = W.shape[0]
        random_indices = np.random.choice(n, size=nSamples, replace=False)
        # return W[random_indices]
        return random_indices


    def forward(self, features, gt, d_size=1,min_points=200, max_points=1000,
                         point_jump=50, flag =1):
        if flag ==1:  # only sig_error(sig2, sig1_2)
            """
            sig_error(sig2, sig1_2). The key part of the topology autoencoder
            """
            samples = random.sample(range(0, features.size()[0] - 1), max_points)
            gt = gt[samples, :]
            features = features[samples, :]

            distances1 = self._compute_distance_matrix(features)
            distances2 = self._compute_distance_matrix(gt)

            pairs2 = self._get_pairings(distances2)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            sig1_2 = self._select_distances_from_pairs(distances1, pairs2)

            m = self.sig_error(sig2, sig1_2)
            return [m, m * 0]

        if flag ==2: # only phd dimention (L_d)
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
                lengths.append(d/d2)

            lengths = torch.stack(lengths)

            # compute our ph dim by running a linear least squares
            x = torch.tensor(test_n).to(lengths).log()
            y = lengths.log()
            N = len(x)
            phd_m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
            phd_m = torch.abs(phd_m)
            return [phd_m * 0, phd_m]

        if flag ==3:  #  phd dimention (L_d) + sig_error(sig2, sig1_2)
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
                lengths.append(d/d2)

            lengths = torch.stack(lengths)

            # compute our ph dim by running a linear least squares
            x = torch.tensor(test_n).to(lengths).log()
            y = lengths.log()
            N = len(x)
            phd_m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
            phd_m = torch.abs(phd_m)

            """
            sig_error(sig2, sig1_2). The key part of the topology autoencoder
            """
            samples = random.sample(range(0, features.size()[0] - 1), max_points)
            gt = gt[samples, :]
            features = features[samples, :]

            distances1 = self._compute_distance_matrix(features)
            distances2 = self._compute_distance_matrix(gt)


            pairs2 = self._get_pairings(distances2)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            sig1_2 = self._select_distances_from_pairs(distances1, pairs2)


            m = self.sig_error(sig2, sig1_2)
            return [m, phd_m]

        if flag ==4:  #  phd dimention (L_d) + autoencoder
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
                lengths.append(d/d2)

            lengths = torch.stack(lengths)

            # compute our ph dim by running a linear least squares
            x = torch.tensor(test_n).to(lengths).log()
            y = lengths.log()
            N = len(x)
            phd_m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
            phd_m = torch.abs(phd_m)

            """
            # Topology Autoencoder 
            # """
            samples = random.sample(range(0, features.size()[0] - 1), max_points)
            gt = gt[samples, :]
            features = features[samples, :]

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

        if flag ==5:  # the original phd dimension (L'_d)
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
                lengths.append(d)
            lengths = torch.stack(lengths)

            # compute our ph dim by running a linear least squares
            x = torch.tensor(test_n).to(lengths).log()
            y = lengths.log()
            N = len(x)
            phd_m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)

            return [phd_m * 0, phd_m]

        if flag ==6:  # Topology autoencoder (L_t)
            """
            Topology Autoencoder 
            """
            samples = random.sample(range(0, features.size()[0] - 1), max_points)
            gt = gt[samples, :]
            features = features[samples, :]

            distances1 = self._compute_distance_matrix(features)
            distances2 = self._compute_distance_matrix(gt)

            pairs1 = self._get_pairings(distances1)
            pairs2 = self._get_pairings(distances2)
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)

            sig1_2 = self._select_distances_from_pairs(distances1, pairs2)
            sig2_1 = self._select_distances_from_pairs(distances2, pairs1)

            m = self.sig_error(sig2, sig1_2) + self.sig_error(sig1, sig2_1)
            return [m, m * 0]

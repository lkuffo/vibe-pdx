from pdxearch.index_factory import IndexPDXIMISQ8
import numpy as np
import math

from ..base.module import BaseANN


class PDXIVF(BaseANN):

    def __init__(self, metric, n_clusters_factor, train_frac):
        self.metric = metric
        self.n_clusters_factor = n_clusters_factor
        self.train_frac = train_frac

    def fit(self, X):
        n_samples, dim = X.shape

        n_clusters = self.n_clusters_factor * math.ceil(math.sqrt(n_samples))

        print("n_clusters:", n_clusters)
        # norms = np.linalg.norm(X, axis=1, keepdims=True)
        # norms[norms == 0] = 1 # Prevent bug if a vector is full of 0's
        # X /= norms

        self.index = IndexPDXIMISQ8(ndim=dim, nbuckets=n_clusters, normalize=True)
        print('Preprocessing')
        train = self.index.preprocess(X, inplace=False)

        print('Training')
        if self.train_frac < 1.0:
            training_points = math.floor(n_samples * self.train_frac)
            rng = np.random.default_rng()
            training_sample_idxs = rng.choice(len(train), size=training_points, replace=False)
            training_sample_idxs.sort()
            self.index.train(train[training_sample_idxs])
        else:
            self.index.train(train)

        print('Adding')
        self.index.add_load(train)


    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        self.fit(X_train)

    def set_query_arguments(self, query_args):
        self.clusters_to_search, self.pruning_confidence = query_args
        self.index.set_pruning_confidence(self.pruning_confidence)

    def query(self, q, n):
        I, D = self.index.search(q, n, nprobe=self.clusters_to_search)
        print(I)
        return I

    def __str__(self):
        str_template = "PDXIVF(nc=%.2f * sqrt(n), cs=%d, pruning_confidence=%d)"
        return str_template % (
            self.n_clusters_factor,
            self.clusters_to_search,
            self.pruning_confidence,
        )

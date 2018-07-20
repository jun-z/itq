import numpy as np


class PCA:
    def __init__(self, num_bits):
        self.num_bits = num_bits

    def fit(self, X):
        num_rows, num_cols = X.shape

        # Center data.
        for i in range(num_cols):
            X[:, i] -= X[:, i].mean()

        eigvals, eigvecs = np.linalg.eig(np.dot(X.T, X))

        # Sort the eigenvalues and eigenvectors.
        sort_indices = np.argsort(-eigvals)

        eigvals = eigvals[sort_indices]
        eigvecs = eigvecs[:, sort_indices]

        self.W = eigvecs[:, :self.num_bits]

    def transform(self, X):
        return np.dot(X, self.W)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

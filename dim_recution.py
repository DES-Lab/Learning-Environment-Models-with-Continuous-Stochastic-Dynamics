from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from autoencoder import AE


def get_observations_and_actions(traces, include_reward=False):
    observation_space = []
    actions = []
    for trace in traces:
        for x in trace:
            state = list(x[0][0])
            if include_reward:
                state.append(x[2])
            actions.append(x[1])
            observation_space.append(state)

    return np.array(observation_space), np.array(actions)


class DimReduction(ABC, BaseEstimator, TransformerMixin):

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X, y=None):
        pass


class ManualLunarLanderDimReduction(DimReduction):
    def __init__(self, env_name):
        self.env_name = env_name

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        transformed = []
        for x in X:
            if self.env_name == 'LunarLander-v2':
                abstracted = [x[0] + x[2], x[1] + x[3], x[4] + x[5], x[6] + x[7]]
                transformed.append(abstracted)
            else:
                raise NotImplementedError(f'Manual dimensionality reduction for {self.env_name} not implemented.')
        return np.array(transformed)


class LdaDimReduction(DimReduction):
    def __init__(self):
        self.lda = LinearDiscriminantAnalysis()

    # y are actions
    def fit(self, X, y=None):
        self.lda.fit(X, y)

    def transform(self, X, y=None):
        return self.lda.transform(X)


class PcaDimReduction(DimReduction):
    def __init__(self, n_dim):
        self.n_dim = n_dim
        self.pca = PCA(n_components=n_dim)

    def fit(self, X, y=None):
        self.pca.fit(X)

    def transform(self, X, y=None):
        return self.pca.transform(X)


class AutoencoderDimReduction(DimReduction):
    def __init__(self, latent_dim, num_training_epochs):
        self.latent_dim = latent_dim
        self.num_training_epochs = num_training_epochs
        self.ae = AE(latent_dim)

    def fit(self, X, y=None):
        self.ae.train_autoencoder(X, epochs=20, batch_size=64)

    def transform(self, X, y=None):
        return np.array([self.ae.encoder(torch.tensor(obs)).detach().numpy().tolist() for obs in X])

import os
import random

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from autoencoder import AE
from utils import load, save


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


class PipelineWrapper(Pipeline):
    def __init__(self, env_name, num_traces, steps, load_pipeline=True):
        super().__init__(steps)
        self.env_name = env_name
        self.num_traces = num_traces
        self.load_pipeline = load_pipeline
        if self.steps:
            self.pipeline_name = f'{self.env_name}_num_traces_{self.num_traces}_' + '_'.join([i[0] for i in steps])
        else:
            self.pipeline_name = 'no_pipeline'
        self.save_path = f'pickles/dim_reduction/{self.pipeline_name}.pk'

    def fit(self, X, y=None, **fit_params):
        if not self.steps:
            return
        if self.load_pipeline and os.path.exists(self.save_path):
            self.steps = load(self.save_path)
            print('Pipeline loaded.')
            assert self.steps
            return self.steps

        super(PipelineWrapper, self).fit(X, y, **fit_params)
        save(self.steps, self.save_path)

    def transform(self, X):
        if not self.steps:
            return X
        return super(PipelineWrapper, self).transform(X)


class AutoencoderDimReduction(BaseEstimator, TransformerMixin):
    def __init__(self, latent_dim, num_training_epochs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_training_epochs = num_training_epochs
        self.ae = AE(latent_dim)

    def fit(self, X, y=None):
        self.ae.train_autoencoder(X, epochs=self.num_training_epochs, batch_size=64)

    def transform(self, X, y=None):
        return np.array([self.ae.encoder(torch.tensor(obs)).detach().numpy().tolist() for obs in X])


def get_k_means_clustering(observations, n_clusters, dim_red_pipeline_name, reduced_samples=None, load_fun=True):
    cf_name = f'pickles/clustering/k_means_num_clusters_{n_clusters}_{dim_red_pipeline_name}.pk'
    if load_fun and os.path.exists(cf_name):
        clustering_function = load(cf_name)
        print(f'K-means with {n_clusters} loaded.')
        assert clustering_function
        cluster_labels = clustering_function.predict(np.array(observations))
        return clustering_function, cluster_labels

    print(f'Computing k-means with {n_clusters} clusters.')
    clustering_function = KMeans(n_clusters=n_clusters, init="k-means++")

    if reduced_samples:
        reduced_obs_space = random.choices(observations, k=reduced_samples)
    else:
        reduced_obs_space = observations

    clustering_function.fit(reduced_obs_space)
    cluster_labels = clustering_function.predict(np.array(observations))

    save(clustering_function, cf_name)
    return clustering_function, cluster_labels

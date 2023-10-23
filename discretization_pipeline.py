import os
import random

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

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


def change_features_lunar_lander(x):
    if x.shape == (8,):
        x = np.expand_dims(x, axis=0)
    transformed = np.zeros(4)
    # transformed[:x.shape[0], :x.shape[1]] = x
    transformed[0] = x[:, 0] + x[:, 2]
    transformed[1] = x[:, 1] + x[:, 3]
    transformed[2] = x[:, 4] + x[:, 5]
    transformed[3] = x[:, 6] + x[:, 7]
    return transformed


def change_features_acrobot(x):
    if x.shape == (6,):
        x = np.expand_dims(x, axis=0)
    transformed = np.zeros(4)

    transformed[0] = np.arctan(x[:, 0] / x[:, 1])
    transformed[1] = np.arctan(x[:, 2] / x[:, 3])
    transformed[2] = x[:, 4]
    transformed[3] = x[:, 5]
    return transformed


class PipelineWrapper(Pipeline):
    def __init__(self, env_name, num_traces, steps, prefix_len=None, load_pipeline=True):
        super().__init__(steps)
        self.env_name = env_name
        self.num_traces = num_traces
        self.load_pipeline = load_pipeline
        self.prefix_len = f'prefix_cut_{prefix_len}_' if prefix_len else ''
        if self.steps:
            self.pipeline_name = f'{self.prefix_len}{self.env_name}_num_traces_{self.num_traces}_' + '_'.join(
                [i[0] for i in steps])
        else:
            self.pipeline_name = f'{env_name}_{num_traces}_no_pipeline'
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


class LunarLanderManualDimReduction(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        return np.array([change_features_lunar_lander(obs).tolist() for obs in X])


class AcrobotManualDimReduction(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        return np.array([change_features_acrobot(obs).tolist() for obs in X])


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

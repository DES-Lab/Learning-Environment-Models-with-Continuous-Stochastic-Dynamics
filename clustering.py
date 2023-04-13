import random

import numpy as np
from sklearn.cluster import KMeans, estimate_bandwidth, MeanShift


def get_k_means_clustering(observations, n_clusters, num_samples=None):
    clustering_function = KMeans(n_clusters=n_clusters, init="k-means++")
    if num_samples:
        reduced_obs_space = random.choices(observations, k=num_samples)
    else:
        reduced_obs_space = observations

    clustering_function.fit(reduced_obs_space)
    cluster_labels = clustering_function.predict(np.array(observations))

    return clustering_function, cluster_labels


def get_mean_shift_clustering(observations, bandwidth_multiplier=1., num_samples=None):
    if num_samples:
        reduced_obs_space = random.choices(observations, k=num_samples)
    else:
        reduced_obs_space = observations

    band_width = estimate_bandwidth(reduced_obs_space) * bandwidth_multiplier
    clustering_function = MeanShift(bandwidth=band_width)
    clustering_function.fit(reduced_obs_space)

    print(f"Mean shift found {len(clustering_function.cluster_centers_)} clusters.")
    cluster_labels = clustering_function.predict(observations)

    return clustering_function, cluster_labels

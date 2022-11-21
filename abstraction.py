import random

import sklearn
from tqdm import tqdm

import numpy as np
from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import estimate_bandwidth
from utils import save


def compute_clustering_function_and_map_to_traces(traces_obtained_from_all_agents,
                                                  action_map,
                                                  n_clusters=16,
                                                  scale=False,
                                                  reduce_dimensions=False,
                                                  clustering_type = "k_means",
                                                  ms_samples = 10000,
                                                  ms_bw_multiplier = 0.5,
                                                  include_reward_in_output=False):
    observation_space = []
    for sampled_data in traces_obtained_from_all_agents:
        for trace in sampled_data:
            for x in trace:
                state = x[0]
                observation_space.append(state[0])
                # observation_space.extend([x[0][0:6] for trace in sampled_data for x in trace])

    num_traces = sum([len(x) for x in traces_obtained_from_all_agents])

    observation_space = np.array(observation_space)
    observation_space = np.squeeze(observation_space)
    scaler = StandardScaler()
    scaler.fit(observation_space)
    save(scaler, f'standard_scaler_{num_traces}')

    if reduce_dimensions:
        pca = PCA(n_components=4)
        observation_space = pca.fit_transform(observation_space)

        save(pca, 'pca_4')
        print('Dimensions reduced with PCA')

    if scale:
        observation_space = scaler.transform(observation_space)

    clustering_function = None
    cluster_labels = None
    if clustering_type == "k_means":
        clustering_function = KMeans(n_clusters=n_clusters)
        clustering_function.fit(observation_space)
        cluster_labels = list(clustering_function.labels_)
    elif clustering_type == "mean_shift":
        # reduced_obs_space_bw = random.choices(observation_space,k=20000)
        reduced_obs_space = random.choices(observation_space,k=ms_samples)
        print("About to estimate bw")
        band_width = estimate_bandwidth(reduced_obs_space) * ms_bw_multiplier
        print("Estimated bw")
        clustering_function = MeanShift(bandwidth=band_width)
        clustering_function.fit(reduced_obs_space)
        print(f"Found {len(clustering_function.cluster_centers_)} clusters with mean shift")
        cluster_labels = clustering_function.predict(observation_space)

    save(clustering_function, f'{clustering_type}_scale_{scale}_{n_clusters}_{num_traces}')

    print('Cluster labels computed')

    alergia_datasets = []

    label_i = 0
    print(f'Creating Alergia Samples. x {len(traces_obtained_from_all_agents)}')
    for policy_samples in traces_obtained_from_all_agents:
        dataset = []
        for sample in tqdm(policy_samples):
            alergia_sample = ['INIT']
            for _, action, reward, done in sample:
                cluster_label = f'c{cluster_labels[label_i]}'
                label_i += 1
                # if include_reward_in_output:
                # cluster_label += f'_{round(reward, 2)}'
                if reward == 100 and done:
                    alergia_sample.append(
                        (action_map[int(action)], f"succ__pos__{cluster_label}"))
                elif reward == -100 and done:
                    alergia_sample.append(
                        (action_map[int(action)], f"bad__{cluster_label}"))
                elif reward >= 10 and done:
                    alergia_sample.append(
                        (action_map[int(action)], f"pos__{cluster_label}"))
                else:
                    alergia_sample.append(
                        (action_map[int(action)], cluster_label if not done else 'DONE'))  # action_map[int(action)]

            dataset.append(alergia_sample)
        alergia_datasets.append(dataset)

    print('Cluster labels replaced')
    return alergia_datasets

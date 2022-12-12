import random

import sklearn
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

import numpy as np
from sklearn.cluster import KMeans, MeanShift, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer, Normalizer
from sklearn.cluster import estimate_bandwidth
from utils import save


def compute_clustering_function_and_map_to_traces(traces_obtained_from_all_agents,
                                                  action_map, env_name,
                                                  n_clusters=16,
                                                  scale=False,
                                                  reduce_dimensions=False,
                                                  clustering_type = "k_means",
                                                  clustering_samples = 100000,
                                                  ms_bw_multiplier = 0.25,
                                                  include_reward = False,
                                                  include_reward_in_output=False):
    observation_space = []
    for sampled_data in traces_obtained_from_all_agents:
        for trace in sampled_data:
            for x in trace:
                state = list(x[0][0])
                reward = x[2]
                if include_reward:
                    state.append(reward)

                observation_space.append(state)
                # observation_space.extend([x[0][0:6] for trace in sampled_data for x in trace])

    num_traces = sum([len(x) for x in traces_obtained_from_all_agents])

    observation_space = np.array(observation_space)
    observation_space = np.squeeze(observation_space)
    # scaler = PowerTransformer()
    # scaler.fit(observation_space)
    # save(scaler, f'power_scaler_{env_name}_{num_traces}')
    scaler = make_pipeline(PowerTransformer(standardize=False))
    scaler.fit(observation_space)
    save(scaler, f'pipeline_scaler_{env_name}_{num_traces}')

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
        clustering_function = KMeans(n_clusters=n_clusters,init="k-means++")
        reduced_obs_space = random.choices(observation_space, k=clustering_samples)
        clustering_function.fit(reduced_obs_space)
        cluster_labels = clustering_function.predict(np.array(observation_space,dtype=float)) #list(clustering_function.labels_)
    elif clustering_type == "mean_shift":
        # reduced_obs_space_bw = random.choices(observation_space,k=20000)
        reduced_obs_space = random.choices(observation_space, k=clustering_samples)
        print("About to estimate bw")
        band_width = estimate_bandwidth(reduced_obs_space) * ms_bw_multiplier
        print("Estimated bw")
        clustering_function = MeanShift(bandwidth=band_width)
        clustering_function.fit(reduced_obs_space)
        print(f"Found {len(clustering_function.cluster_centers_)} clusters with mean shift")
        cluster_labels = clustering_function.predict(observation_space)

    save(clustering_function, f'{env_name}_{clustering_type}_scale_{scale}_{n_clusters}_{num_traces}')

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

                if "Lunar" in env_name and  reward == 100 and done:
                    alergia_sample.append(
                        (action_map[int(action)], f"{cluster_label}__succ__pos"))
                elif "Lunar" in env_name and reward == -100 and done:
                    alergia_sample.append(
                        (action_map[int(action)], f"{cluster_label}__bad"))
                elif "Lunar" in env_name and reward >= 10 and done:
                    alergia_sample.append(
                        (action_map[int(action)], f"{cluster_label}__pos"))
                elif "Lunar" in env_name and reward >= 10 and done:
                    alergia_sample.append(
                        (action_map[int(action)], f"{cluster_label}__pos"))
                elif "Mountain" in env_name and done and len(alergia_sample) < 200:
                    alergia_sample.append(
                        (action_map[int(action)], f"{cluster_label}__succ"))
                else:
                    alergia_sample.append(
                        (action_map[int(action)], cluster_label if not done else 'DONE'))  # action_map[int(action)]

            dataset.append(alergia_sample)
        alergia_datasets.append(dataset)

    print('Cluster labels replaced')
    return alergia_datasets

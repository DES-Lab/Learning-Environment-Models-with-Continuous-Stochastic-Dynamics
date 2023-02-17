import random

import sklearn
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

import numpy as np
from sklearn.cluster import KMeans, MeanShift, MiniBatchKMeans
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.preprocessing import StandardScaler, PowerTransformer, Normalizer, FunctionTransformer
from sklearn.cluster import estimate_bandwidth

from utils import save
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def change_features_clustering(x):
    transformed = np.zeros((x.shape[0],4))
    # transformed[:x.shape[0], :x.shape[1]] = x
    transformed[:, 0] = x[:, 0] + x[:,2]
    transformed[:, 1] = x[:, 1] + x[:,3]
    transformed[:, 2] = x[:, 4] + x[:,5]
    transformed[:, 3] = x[:, 6] + x[:,7]
    return transformed

class LDATransformer:
    def __init__(self,x,y):
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(x, y)

    def fit(self,x,y):
        pass
    def transform(self,x):
        return self.lda.transform(x)

# def create_lda_estimator(x,y):
#     lda = LinearDiscriminantAnalysis()
#     lda.fit(x,y)
#     return lambda x : lda.transform(x)

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
    actions = []
    for sampled_data in traces_obtained_from_all_agents:
        for trace in sampled_data:
            for x in trace:
                # print(x)
                state = list(x[0][0])
                # reward = x[2]
                # if include_reward:
                #     state.append(reward)
                actions.append(x[1])
                observation_space.append(state)
                # observation_space.extend([x[0][0:6] for trace in sampled_data for x in trace])

    num_traces = sum([len(x) for x in traces_obtained_from_all_agents])

    observation_space = np.array(observation_space)
    observation_space = np.squeeze(observation_space)
    actions = np.array(actions)
    actions = np.squeeze(actions)
    # scaler = PowerTransformer()
    # scaler.fit(observation_space)
    # save(scaler, f'power_scaler_{env_name}_{num_traces}')

    # scaler = make_pipeline(FunctionTransformer(change_features_clustering),PowerTransformer())
    # scaler = make_pipeline(PowerTransformer(), PCA(n_components=4))

    # scaler = make_pipeline(PCA(n_components=6))
    # scaler = make_pipeline(LDATransformer(observation_space,actions))

    # this works
    scaler = make_pipeline(StandardScaler(), LDATransformer(observation_space, actions))

    # scaler = make_pipeline(PCA(n_components=4),PowerTransformer())

    scaler.fit(observation_space)
    save(scaler, f'power_scaler_{env_name}_{num_traces}')

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
            last_cluster = None
            for state, action, reward, done in sample:
                cluster_label_int = cluster_labels[label_i]
                next_cluster_label_int = cluster_labels[label_i+1] if label_i + 1 < len(cluster_labels) else None
                cluster_label = f'c{cluster_label_int}'
                label_i += 1
                # if include_reward_in_output:
                # cluster_label += f'_{round(reward, 2)}'

                # additional_label = "__low" if "Lunar" in env_name and state[0][1] <= 0.2 else ""
                # if "Lunar" in env_name and  reward == 100 and done:
                #     alergia_sample.append(
                #         (action_map[int(action)], f"{cluster_label}__succ__pos{additional_label}"))
                # elif "Lunar" in env_name and reward == -100 and done:
                #     alergia_sample.append(
                #         (action_map[int(action)], f"{cluster_label}__bad{additional_label}"))
                # elif "Lunar" in env_name and reward >= 10 and done:
                #     alergia_sample.append(
                #         (action_map[int(action)], f"{cluster_label}__pos{additional_label}"))
                # elif "Lunar" in env_name and reward >= 10 and done:
                #     alergia_sample.append(
                #         (action_map[int(action)], f"{cluster_label}__pos{additional_label}"))
                # elif "Mountain" in env_name and done and len(alergia_sample) < 200 and state[0][0] > 0:
                #     alergia_sample.append(
                #         (action_map[int(action)], f"{cluster_label}__succ"))
                # else:
                #     alergia_sample.append(
                #         (action_map[int(action)], f"{cluster_label}{additional_label}" if not done else 'DONE'))

                action = action_map[int(action)]
                label_string = create_label(len(alergia_sample), cluster_label, cluster_label_int, done, env_name,
                                            last_cluster, next_cluster_label_int, reward, state)
                last_cluster = cluster_label_int
                alergia_sample.append((action,label_string))

            dataset.append(alergia_sample)
        alergia_datasets.append(dataset)

    print('Cluster labels replaced')
    return alergia_datasets


def create_label(nr_steps, cluster_label, cluster_label_int, done, env_name, last_cluster, next_cluster_label_int,
                 reward, state):
    additional_labels = []
    if "Lunar" in env_name and reward == 100 and done:
        additional_labels.append("succ")
        additional_labels.append("pos")
    elif "Lunar" in env_name and reward == -100 and done:
        additional_labels.append("bad")
    elif "Lunar" in env_name and reward >= 10 and done:
        additional_labels.append("pos")
    elif "Mountain" in env_name and done and nr_steps < 200 and state[0][0] > 0:
        additional_labels.append("succ")
    elif done:
        additional_labels.append("DONE")

    # if "Lunar" in env_name and state[0][1] < 0.1:
    #     additional_labels.append("low")
    # if "Lunar" in env_name and abs(state[0][0]) < 0.5:
    #     additional_labels.append("mid")
    # if last_cluster != cluster_label_int:
    #     additional_labels.append("entry")
    # if cluster_label_int != next_cluster_label_int:
    #     additional_labels.append("exit")
    labels = [cluster_label] + additional_labels
    label_string = "__".join(labels)
    return label_string

import os
import pickle

from aalpy.base import SUL
from sklearn.cluster import KMeans


class GymSUL(SUL):
    def __init__(self, env, clustering_fun):
        super().__init__()
        self.env = env
        self.clustering_fun = clustering_fun
        self.last_obs = None

    def pre(self):
        obs = self.env.reset()
        self.last_obs = obs
        return obs

    def post(self):
        pass

    def step(self, letter):
        if letter is None:
            return self.last_obs
        obs, reward, done, _ = self.env.step(letter)
        cluster = self.clustering_fun.predict(obs.reshape(1, -1))
        return f'c{cluster}' if not done else 'DONE'


def compute_clusters(data, n_clusters):
    clustering_function = KMeans(n_clusters=n_clusters)
    clustering_function.fit(data)
    return clustering_function


def save(x, path):
    with open(f'{path}.pickle', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(load_path):
    if os.path.exists(load_path):
        with open(load_path, 'rb') as handle:
            return pickle.load(handle)
    else:
        return None


def save_samples_to_file(samples, filename='jAlergiaData.txt'):
    with open(filename, 'w') as f:
        for seq in samples:
            f.write(','.join([str(s) for s in seq]) + '\n')


def delete_file(filename):
    import os
    if os.path.exists(filename):
        os.remove(filename)

# def old_stuff():
# num_episodes = 10000
# num_clusters = 32
#
# sampled_data = []
#
# # sample data
# observation, info = env.reset()
#
# for _ in range(num_episodes):
#     episode_trace = []
#     observation, info = env.reset()
#
#     while True:
#         action = env.action_space.sample()
#         observation, reward, terminated, truncated, _ = env.step(action)
#
#         episode_trace.append((observation.reshape(1, -1), action, reward))
#         if terminated or truncated:
#             break
#
#     sampled_data.append(episode_trace)
#
# env.close()
#
# # cluster over observation space
# observation_space = [x[0] for trace in sampled_data for x in trace]
#
# observation_space = np.array(observation_space)
# print(observation_space.shape)
# observation_space = np.squeeze(observation_space)
# # num_samples, nx, ny = observation_space.shape
# # observation_space = observation_space.reshape((num_samples, nx*ny))
# # print(observation_space.shape)
#
# clustering_function = compute_clusters(observation_space, num_clusters)
#
# # print('CF computed')
# # for i in observation_space[:10]:
# #     print(clustering_function.predict(i.reshape(1, -1)))
#
# # active learning
# alphabet = list(range(env.action_space.n))
# sul = GymSUL(env, clustering_function)
# eq_oracle = RandomWordEqOracle(alphabet, sul, min_walk_len=5, max_walk_len=30)
#
# model = run_stochastic_Lstar(alphabet, sul, eq_oracle, automaton_type='mdp', max_rounds=15)
#
# model.save()
# model.visualize()

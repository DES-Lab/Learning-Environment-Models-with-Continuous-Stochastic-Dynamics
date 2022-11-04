import gym
import numpy as np
from aalpy.learning_algs import run_Alergia
from aalpy.utils import statistical_model_checking
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from stable_baselines3 import A2C, DQN

from agents import load_agent


# action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}


def get_traces_from_policy(agent, env, num_episodes):
    traces = []
    for _ in range(num_episodes):
        observation = env.reset()

        episode_trace = []
        while True:
            action, _ = agent.predict(observation)
            observation, reward, done, info = env.step(action)

            episode_trace.append((observation.reshape(1, -1), action, reward, done))
            if done:
                break

        traces.append(episode_trace)

    return traces


def compute_clustering_function(all_policy_traces, n_clusters=16, reduce_dimensions=False):
    observation_space = []
    for sampled_data in all_policy_traces:
        observation_space.extend([x[0] for trace in sampled_data for x in trace])

    observation_space = np.array(observation_space)
    observation_space = np.squeeze(observation_space)

    pca = None
    if reduce_dimensions:
        pca = PCA(n_components=16)
        pca.fit_transform(observation_space)
        print('PCA trained')

    clustering_function = KMeans(n_clusters=n_clusters)
    clustering_function.fit(observation_space)
    return clustering_function, pca


def compute_stochastic_model(traces, clustering_fun, pca=None):
    alergia_traces = []
    for trace in traces:
        alergia_trace = ['INIT']
        for obs, action, reward, done in trace:
            if pca:
                obs = pca.transform(obs.reshape(1, -1))
            cluster = f'C{clustering_fun.predict(obs.reshape(1, -1))[0]}'
            alergia_trace.append((int(action), cluster if not done else 'DONE'))  # action_map[int(action)]

        alergia_traces.append(alergia_trace)

    model = run_Alergia(alergia_traces, automaton_type='mdp', print_info=True)
    model.make_input_complete()
    return model


environment = "LunarLander-v2"

env = gym.make(environment, )

dqn_agent = load_agent('araffin/ppo-LunarLander-v2', 'ppo-LunarLander-v2.zip', A2C)
a2c_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
print('Agents loaded')

num_traces = 1500
num_clusters = 16

all_data = []
for agent in [dqn_agent, a2c_agent]:
    all_data.append(get_traces_from_policy(agent, env, num_traces))

print('Traces obtained')
clustering_function, pca = compute_clustering_function(all_data, n_clusters=num_clusters, reduce_dimensions=False)
print('Clustering function computed')

mdp_dqn = compute_stochastic_model(all_data[0], clustering_function)
mdp_a2c = compute_stochastic_model(all_data[1], clustering_function)

mdp_dqn.save('mdp_dqn')
mdp_a2c.save('mdp_a2c')

print('Performing SMC')
dqn_smc = statistical_model_checking(mdp_dqn, {'DONE'}, max_num_steps=35)
a2c_smc = statistical_model_checking(mdp_a2c, {'DONE'}, max_num_steps=35)

print(f'DQN: {dqn_smc}\nA2C: {a2c_smc}')

import random
from statistics import mean

import aalpy.paths
import gym
import numpy as np
from aalpy.learning_algs import run_JAlergia
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN
from tqdm import tqdm

from abstraction import LDATransformer
from agents import load_agent
from utils import save_samples_to_file, delete_file


def get_alergia_traces_and_clustering_function(traces,
                                               n_clusters=64,
                                               include_reward=False,
                                               scale=False,
                                               clustering_type="k_means",
                                               clustering_samples=100000, ):
    observation_space = []
    actions = []
    for trace in traces:
        for x in trace:
            state = list(x[0][0])
            # state = state[:2]
            reward = x[2]
            if include_reward:
                state.append(reward)

            observation_space.append(state)
            actions.append(x[1])

    observation_space = np.array(observation_space)
    observation_space = np.squeeze(observation_space)

    clustering_function = None
    cluster_labels = None

    scaler = None
    if scale:
        actions = np.array(actions)
        actions = np.squeeze(actions)
        scaler = make_pipeline(StandardScaler(), LDATransformer(observation_space, actions))
        scaler.fit(observation_space)
        observation_space = scaler.transform(observation_space)

    if clustering_type == "k_means":
        clustering_function = KMeans(n_clusters=n_clusters, )

        reduced_obs_space = observation_space
        if clustering_samples:
            reduced_obs_space = random.choices(observation_space, k=clustering_samples)
        clustering_function.fit(reduced_obs_space)
        cluster_labels = clustering_function.predict(
            np.array(observation_space, dtype=float))  # list(clustering_function.labels_)

    elif clustering_type == "bisecting_k_means":
        pass

    print('Cluster labels computed')
    label_i = 0

    print(f'Creating Alergia Samples.')
    alergia_samples = []
    for sample in tqdm(traces):
        alergia_sample = ['INIT']
        for state, action, reward, done in sample:
            reached_cluster = cluster_labels[label_i]
            label_i += 1

            if done:
                # f'c{cluster_labels[label_i]}_DONE'
                reached_cluster = 'DONE'

            action = action_map[int(action)]
            alergia_sample.append((f'c{reached_cluster}', action))

        alergia_samples.append(alergia_sample)

    return alergia_samples, clustering_function, scaler


def test_model(env, mdp, clustering_function, scaler, num_episodes=10, render=False):
    all_rewards = []
    return

    for _ in range(num_episodes):
        ep_reward = 0
        obs = env.reset()

        mdp.reset_to_initial()

        origin_cluster, reached_cluster = None, None

        while True:
            obs = obs.reshape(1, -1).astype(float)
            if scaler:
                obs = scaler.transform(obs)

            origin_cluster = f'c{clustering_function.predict(obs)[0]}'
            transition_label = f'{origin_cluster}_{reached_cluster}'
            if origin_cluster is None or reached_cluster is None or \
                    transition_label not in mdp.current_state.transitions.keys():
                action = random.choice(list(action_map.keys()))
                # print('Random Action')

                for state in mdp.states:
                    if state.output == action:
                        mdp.current_state = state
                        break

            else:
                outgoing_transitions = mdp.current_state.transitions[transition_label]
                actions = [p[0].output for p in outgoing_transitions]
                probabilities = [p[1] for p in outgoing_transitions]
                action = input_map[random.choices(actions, weights=probabilities, k=1)[0]]

                mdp.step_to(transition_label, action)

            obs, rew, done, info = env.step(action)
            if render:
                env.render()
            reached_cluster = f'c{clustering_function.predict(np.array(obs.reshape(1, -1), dtype=float))[0]}'

            ep_reward += rew
            if done:
                all_rewards.append(ep_reward)
                break

    print(f'Mean episodes reward: {mean(all_rewards)}')


def iterative_model_generation(agent, env, num_iterations=10, episodes_per_iteration=100):
    jalergia_sample_file = 'edi_alergia.txt'
    observations = []
    learned_model = None

    for learning_round in range(num_iterations):

        print('Obtaining traces')
        for _ in range(episodes_per_iteration):
            episode_trace = []
            obs = env.reset()
            while True:
                action, _ = agent.predict(obs)
                observation, reward, done, info = env.step(action)
                episode_trace.append((observation.reshape(1, -1), action, reward, done))

                if done:
                    observations.append(episode_trace)
                    break

        alergia_traces, clustering_function, scaler = get_alergia_traces_and_clustering_function(observations,
                                                                                                 n_clusters=16,
                                                                                                 scale=True)

        save_samples_to_file(alergia_traces, jalergia_sample_file)

        learned_model = run_JAlergia(jalergia_sample_file, 'mdp', 'alergia.jar',
                                     heap_memory='-Xmx6G', optimize_for='accuracy', eps=0.01)
        # learned_model.make_input_complete(missing_transition_go_to='sink_state')
        delete_file(jalergia_sample_file)
        print('Model computed')

        # prism_interface = PrismInterface(['DONE'], learned_model)
        # scheduler = ProbabilisticScheduler(prism_interface.scheduler, True)
        test_model(env, learned_model, clustering_function, scaler, render=True)
    # learned_model.visualize(display_same_state_transitions=False)

if __name__ == '__main__':

    environment = 'LunarLander-v2'
    env = gym.make(environment)

    aalpy.paths.path_to_prism = "C:/Program Files/prism-4.6/bin/prism.bat"

    dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)

    action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
    input_map = {v: k for k, v in action_map.items()}

    iterative_model_generation(dqn_agent, env)

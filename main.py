import random

import gym
import numpy as np
from aalpy.learning_algs import run_JAlergia, run_Alergia
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN

from abstraction import compute_clustering_function_and_map_to_traces
from agents import get_lunar_lander_agents, load_agent
from dim_recution import get_observations_and_actions, ManualLunarLanderDimReduction, AutoencoderDimReduction, \
    PcaDimReduction, LdaDimReduction
from prism_scheduler import ProbabilisticScheduler, PrismInterface, compute_weighted_clusters
from utils import load, save, delete_file, save_samples_to_file, get_traces_from_policy, create_abstract_traces

environment = "LunarLander-v2"
action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}

agents = None
agent_names = None

if environment == 'LunarLander-v2':
    agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
    # get_lunar_lander_agents(evaluate=False)
    # agents = [a[1] for a in agents_and_names]
    # agent_names = '_'.join(a[0] for a in agents_and_names)
else:
    exit(1)

# print('Agents loaded')

num_traces = 2500
num_clusters = 256
scale = 'ae'

env = gym.make(environment, )
traces_file_name = f'{environment}_{num_traces}_traces'

traces = get_traces_from_policy(agent, env, num_episodes=10, )

obs, actions = get_observations_and_actions(traces)

manual = ManualLunarLanderDimReduction('LunarLander-v2')
lda = LdaDimReduction()
pca = PcaDimReduction(n_dim=4)
ae = AutoencoderDimReduction(4, 20)

#pca.fit(obs)

#transformed = pca.transform(obs)

from clustering import get_k_means_clustering, get_mean_shift_clustering

scaler = make_pipeline(StandardScaler(), LdaDimReduction())
scaler.fit(obs, actions)

transformed = scaler.transform(obs)

cf, cl = get_k_means_clustering(transformed, 10)
cf2, cl2 = get_mean_shift_clustering(transformed, 0.5)


data = create_abstract_traces(traces, cl)
model = run_Alergia(data, automaton_type='mdp')

import aalpy.paths

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.6/bin/prism.bat"


class IterativeRefinement:
    def __init__(self, env, initial_model, traces, dim_reduction_pipeline, clustering_fun):
        self.env = env
        self.model = initial_model
        self.traces = traces
        self.dim_reduction_pipeline = dim_reduction_pipeline
        self.clustering_fun = clustering_fun

    def iteratively_refine_model(self, num_iterations, episodes_per_iteration, goal_state='GOAL'):

        nums_goal_reached = 0

        for refinement_iteration in range(num_iterations):
            model.make_input_complete('sink_state')
            scheduler = PrismInterface(goal_state, self.model).scheduler
            scheduler = ProbabilisticScheduler(scheduler, truly_probabilistic=True)

            num_goal_reached_iteration = 0
            num_crashes_per_iteration = 0

            for _ in range(episodes_per_iteration):
                scheduler.reset()
                self.env.reset()
                ep_data = []

                while True:
                    scheduler_input = scheduler.get_input()
                    if scheduler_input is None:
                        print('Could not schedule an action.')
                        break

                    action = np.array(int(scheduler_input[1:]))

                    observation, reward, done, _ = self.env.step(action)

                    abstract_obs = self.dim_reduction_pipeline.transform(np.array([observation]))

                    reached_cluster = self.clustering_fun.predict(abstract_obs)[0]
                    reached_cluster = f'c{abstract_obs}'

                    if reached_cluster == goal_state:
                        num_goal_reached_iteration += 1

                    weighted_clusters = compute_weighted_clusters(abstract_obs, self.clustering_fun,
                                                                  len(set(self.clustering_fun.labels_)))

                    step_successful = scheduler.step_to(scheduler_input, weighted_clusters)
                    if not step_successful:
                        print('Could not step in a model')
                        break

                    ep_data.append((observation.reshape(1, -1), action, reward, done))

                    if done:
                        if reward == 100 and goal_state == 'GOAL':
                            nums_goal_reached += 1
                            print('Landed')
                        if reward == -100:
                            num_crashes_per_iteration += 1
                        break

                self.traces.append(ep_data)

            nums_goal_reached += num_goal_reached_iteration
            print(f'# Goal Reached : {num_goal_reached_iteration} / {episodes_per_iteration}')
            print(f'# Crashes  : {num_crashes_per_iteration} / {episodes_per_iteration}')

            # refine model
            observation_space, action_space = get_observations_and_actions(self.traces)
            reduced_dim_obs_space = self.dim_reduction_pipeline.transform(observation_space)
            cluster_labels = self.clustering_fun.predict(reduced_dim_obs_space)

            extended_data = create_abstract_traces(self.traces, cluster_labels)
            self.model = run_Alergia(extended_data, automaton_type='mdp')
            print(f'Refinement {refinement_iteration + 1} model size: {self.model.size} states')


ir = IterativeRefinement(env, model, traces, scaler, cf, )

ir.iteratively_refine_model(10, 100)

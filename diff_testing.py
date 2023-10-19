import copy
import pickle
import random
import sys
from collections import Counter, defaultdict
from statistics import mean

import aalpy.paths
import gym
import numpy as np
import torch
from scipy.stats import fisher_exact, mannwhitneyu
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.utils import obs_as_tensor
from tqdm import tqdm

import time
from agents import load_agent
from discretization_pipeline import get_observations_and_actions
from iterative_refinement import IterativeRefinement
from schedulers import PrismInterface, ProbabilisticScheduler, compute_weighted_clusters
from trace_abstraction import create_abstract_traces
from utils import load, mdp_from_state_setup, remove_nan, ACROBOT_GOAL, MOUNTAIN_CAR_GOAL, CARTPOLE_CUTOFF

# aalpy.paths.path_to_prism = "C:/Program Files/prism-4.7/bin/prism.bat"
# aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.8-linux64-x86/bin/prism"
# aalpy.paths.path_to_prism = "C:/Program Files/prism-4.7/bin/prism.bat"

aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.8-linux64-x86/bin/prism"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cluster_frequency(abstract_traces):
    cluster_counter = Counter()
    for at in abstract_traces:
        clusters = at[2::2]
        for c in clusters:
            cluster_counter[c] += 1

    return cluster_counter


def get_cluster_importance(dqn_agent, dim_red_pipeline, clustering_function, num_episodes=1000, importance_def = 0):
    randomness_probabilities = (0, 0.05, 0.1, 0.15, 0.2)
    print('Executing agent policy and assigning importance to each cluster.')
    rand_i = 0
    concrete_traces = []
    importance_list = []
    for _ in tqdm(range(num_episodes)):
        curr_randomness = randomness_probabilities[rand_i]

        observation = env.reset()
        episode_trace = []
        step = 0
        while True:
            if random.random() < curr_randomness:
                action = random.randint(0, env.action_space.n - 1)
            else:
                action, _ = dqn_agent.predict(observation)

            observation, reward, done, info = env.step(action)

            with torch.no_grad():
                q_values = dqn_agent.q_net(obs_as_tensor(observation.reshape(1, -1), device=device))
                q_values = q_values.detach().cpu().numpy().tolist()[0]
                importance = max(q_values) - min(q_values)
                importance_list.append(importance)

            if "CartPole" in env.unwrapped.spec.id and step >= CARTPOLE_CUTOFF:
                done = True

            step += 1
            episode_trace.append((observation.reshape(1, -1), action, reward, done))

            if done:
                concrete_traces.append(episode_trace)
                break

    observation_space, action_space = get_observations_and_actions(concrete_traces)
    if dim_red_pipeline is not None:
        reduced_dim_obs_space = dim_red_pipeline.transform(observation_space)
    else:
        reduced_dim_obs_space = observation_space
    cluster_labels = clustering_function.predict(reduced_dim_obs_space)

    abstract_traces = create_abstract_traces(env_name, concrete_traces, cluster_labels)

    cluster_importance_map = defaultdict(list)
    visited_cluster_index = 0
    for trace in abstract_traces:
        clusters = trace[2::2]
        for c in clusters:
            cluster_importance_map[c].append(importance_list[visited_cluster_index])
            visited_cluster_index += 1

    for k, v in cluster_importance_map.items():
        cluster_importance_map[k] = mean(v), min(v), max(v),mean(v)*len(v)
    sorted_dict = dict(sorted(cluster_importance_map.items(), key=lambda x: x[1][importance_def], reverse=True))

    clusters_with_biggest_mean_importance = []
    for k, v in sorted_dict.items():
        if 'close' not in k and 'succ' not in k:
            clusters_with_biggest_mean_importance.append(k)

        if len(clusters_with_biggest_mean_importance) == 10:
            break

    return clusters_with_biggest_mean_importance


def diff_test(f1, n1, f2, n2, eps):
    contingency_table = np.array([[f1, n1 - f1], [f2, n2 - f2]])
    res = fisher_exact(contingency_table, alternative='two-sided')
    if res[1] < eps:
        return True, res
    else:
        return False, res


def agent_suffix(agent, env, env_name, last_observation, ep_steps):
    observation = last_observation
    suffix_steps = 0
    while True:
        action, _ = agent.predict(observation)
        observation, reward, done, _ = env.step(action)

        ep_steps += 1
        suffix_steps += 1

        if done:
            if env_name == 'LunarLander-v2':
                if reward >= 80:
                    return 'Landed'
                if reward < -80:
                    return 'Crash'
                else:
                    return 'Time_out'
            elif env_name == 'Acrobot-v1' or env_name == 'MountainCar-v0':
                return suffix_steps
            elif env_name == 'CartPole-v1':
                if ep_steps >= CARTPOLE_CUTOFF:
                    return 'Pass'
                return 'Fail'
            else:
                assert False


def stop_based_on_statistics(results_dict, env_name):
    # return True to stop early
    agent_1, agent_2 = agents_under_test[0][0], agents_under_test[1][0]

    if env_name == 'LunarLander-v2':
        f1 = results_dict[agent_1]['Crash'] + results_dict[agent_1]['Time_out']
        f2 = results_dict[agent_2]['Crash'] + results_dict[agent_2]['Time_out']

        n1 = results_dict[agent_1]['Landed'] + f1
        n2 = results_dict[agent_2]['Landed'] + f2

        diff, ratio = diff_test(f1, n1, f2, n2, 0.05)
        if n1 > 0 and n2 > 0 and diff:
            print("Stopping early due to significant differance.")
            print(f'{agent_1} vs {agent_2}')
            print(f"{f1 / n1} vs {f2 / n2}")
            return True
        return False

    elif env_name == 'Acrobot-v1' or env_name == 'MountainCar-v0':
        agent_1_num_steps = results_dict[agent_1]
        agent_2_num_steps = results_dict[agent_2]
        if len(agent_1_num_steps) > 0 and len(agent_2_num_steps) > 0:
            _, p = mannwhitneyu(agent_1_num_steps, agent_2_num_steps)
            if p < 0.05:
                print("Stopping early due to significant differance.")
                print(f'{agent_1} vs {agent_2}')
                print(f'p value: {p}')
                return True
        return False

    elif env_name == 'CartPole-v1':
        f1 = results_dict[agent_1]['Fail']
        f2 = results_dict[agent_2]['Fail']

        n1 = results_dict[agent_1]['Pass'] + f1
        n2 = results_dict[agent_2]['Pass'] + f2

        diff, ratio = diff_test(f1, n1, f2, n2, 0.05)
        if n1 > 0 and n2 > 0 and diff:
            print("Stopping early due to significant differance.")
            print(f'{agent_1} vs {agent_2}')
            print(f"{f1 / n1} vs {f2 / n2}")
            return True
    else:
        assert False


def test_agents(env, env_name, model, agents_under_test, dim_reduction_pipeline, clustering_fun,
                target_clusters, num_tests_per_agent=100, allowed_spurious_ration=0.95, verbose=True):
    assert len(agents_under_test) == 2

    if env_name == 'LunarLander-v2' or env_name == 'CartPole-v1':
        test_results_per_agent = defaultdict(Counter)
    else:
        test_results_per_agent = defaultdict(list)

    # due to a bug in alergia
    remove_nan(model)
    # Make model input complete
    model.make_input_complete('sink_state')

    scheduler = PrismInterface(target_clusters, model).scheduler
    if scheduler is None:
        print('Scheduler could not be computed.')
        assert False

    scheduler = ProbabilisticScheduler(scheduler, truly_probabilistic=True)

    total_test_num, spurious_tests_num, num_successful_tests = 0, 0, 0

    agent_time = 0
    while num_successful_tests < num_tests_per_agent:
        print("Tests: ", total_test_num, spurious_tests_num, num_successful_tests)

        stop_early_based_on_significant_differance = stop_based_on_statistics(test_results_per_agent, env_name)
        if stop_early_based_on_significant_differance:
            break

        if total_test_num > 0 and total_test_num > 10 and spurious_tests_num / total_test_num > allowed_spurious_ration:
            # retrain the model
            print('Number of spurious test cases passed allowed successful/spurious ration.')
            print('Retrain the model')
            timing_info["agent"].append(agent_time)
            return False, test_results_per_agent

        for agent_name, agent in agents_under_test:
            # when true, execute tests with other agent
            switch_agent = False
            while True:
                if total_test_num > 0 and total_test_num > 10 and spurious_tests_num / total_test_num > allowed_spurious_ration:
                    break
                total_test_num += 1

                scheduler.reset()
                env.reset()
                ep_steps = 0

                while True:

                    scheduler_input = scheduler.get_input()
                    if scheduler_input is None:
                        print('Could not schedule an action.')
                        break

                    action = np.array(int(scheduler_input[1:]))

                    observation, reward, done, _ = env.step(action)

                    ep_steps += 1

                    if dim_reduction_pipeline is not None:
                        abstract_obs = dim_reduction_pipeline.transform(np.array([observation]))
                    else:
                        abstract_obs = [observation.reshape(1, -1)]

                    reached_cluster = np.argmin(
                        clustering_fun.transform(abstract_obs))  # clustering_fun.predict(abstract_obs)[0]
                    reached_cluster = f'c{reached_cluster}'

                    if reached_cluster in target_clusters:
                        agent_suffix_start = time.time()
                        test_result = agent_suffix(agent, env, env_name, observation, ep_steps)
                        agent_suffix_end = time.time()
                        agent_time += (agent_suffix_end - agent_suffix_start)

                        if env_name == 'LunarLander-v2' or env_name == 'CartPole-v1':
                            test_results_per_agent[agent_name][test_result] += 1
                        else:
                            test_results_per_agent[agent_name].append(test_result + ep_steps)

                        num_successful_tests += 1
                        switch_agent = True
                        break

                    weighted_clusters = compute_weighted_clusters(scheduler, abstract_obs, scheduler_input,
                                                                  clustering_fun,
                                                                  len(set(clustering_fun.labels_)))

                    step_successful = scheduler.step_to(scheduler_input, weighted_clusters)

                    if not step_successful or done:
                        # Done has been reached before the model could reach the cluster
                        spurious_tests_num += 1
                        break

                if switch_agent:
                    break

    timing_info["agent"].append(agent_time)
    print('Printing experiment results')
    print(f'Total number of tests    : {total_test_num}')
    print(f'Number of spurious tests : {spurious_tests_num}')
    print(f'Percentage valid tests   : {round(100 - (spurious_tests_num / total_test_num) * 100, 2)}%')

    for agent_name, test_res in test_results_per_agent.items():
        if env_name == 'Acrobot-v1' or env_name == 'MountainCar-v0':
            print(agent_name, test_res)
        else:
            print(f'{agent_name}----------------')
            for k, v in test_res.items():
                print(f'{k} : {v}')

    return True, test_results_per_agent


# Load data from experiment run
experiment_data_path = 'pickles/results/lda_mexp1_LunarLander-v2_num_traces_2500_lda_powerTransformer_n_clusters_1024_ri_25_ep_50.pk'
# experiment_data_path = 'pickles/results/cp_64_4_CartPole-v1_num_traces_2500_powerTransformer_n_clusters_64_ri_15_ep_50.pk'
# experiment_data_path = 'pickles/results/mc_64_exp_0_MountainCar-v0_num_traces_2500_powerTransformer_n_clusters_64_ri_25_ep_50.pk'
# experiment_data_path = 'pickles/results/A_exp0_Acrobot-v1_num_traces_2500_manualMapper_n_clusters_256_ri_25_ep_50.pk'
# experiment_data_path = 'pickles/results/ac_alt_lda_exp_0_Acrobot-v1_num_traces_2500_lda_alt3_powerTransformer_lda_alt_n_clusters_512_ri_25_ep_50.pk'

experiment_data = load(experiment_data_path)

last_iter_index = max(list(experiment_data.keys()))
abstract_traces = experiment_data[0]['learning_data']
model = mdp_from_state_setup(experiment_data[last_iter_index]['model'])
dim_red_pipeline = experiment_data[0]['dim_red_pipeline']
clustering_fun = experiment_data[0]['clustering_function']

cluster_counter = get_cluster_frequency(abstract_traces)

cluster_smallest_frequency = [x[0] for x in cluster_counter.most_common() if 'succ' not in x[0]]
cluster_smallest_frequency.reverse()

clusters_of_interest = cluster_smallest_frequency[:10]

env_name = 'LunarLander-v2'
env = gym.make(env_name, )

timing_info = defaultdict(list)
ir = IterativeRefinement(env=env, env_name=env_name, abstract_traces=abstract_traces,
                         initial_model=model,
                         dim_reduction_pipeline=dim_red_pipeline,
                         clustering_fun=clustering_fun,
                         experiment_name_prefix='test_diff',timing_info=timing_info)

ir.current_iteration = last_iter_index + 1
ir.results = experiment_data
# model = ir.model

agents_under_test = []
if "Lunar" in experiment_data_path:
    agents_under_test.extend([
        # these two have similar performance
        ('araffin/dqn-LunarLander-v2', load_agent('araffin/dqn-LunarLander-v2',
                                                  'dqn-LunarLander-v2.zip', DQN)),
        ('araffin/ppo-LunarLander-v2', load_agent('araffin/ppo-LunarLander-v2',
                                                  'ppo-LunarLander-v2.zip', PPO)),
        # ('araffin/a2c-LunarLander-v2', load_agent('araffin/a2c-LunarLander-v2',
        #                                           'a2c-LunarLander-v2.zip', A2C)),
        # ('sb3/dqn-LunarLander-v2', load_agent('sb3/dqn-LunarLander-v2',
        #                                       'dqn-LunarLander-v2.zip', DQN)),
        # ('sb3/a2c-LunarLander-v2', load_agent('sb3/a2c-LunarLander-v2',
        #                                       'a2c-LunarLander-v2.zip', A2C)),
        # ('sb3/ppo-LunarLander-v2', load_agent('sb3/ppo-LunarLander-v2',
        #                                       'ppo-LunarLander-v2.zip', PPO)),
    ])
if "CartPole" in experiment_data_path:
    agents_under_test.extend([
        ('sb3/dqn-CartPole-v1', load_agent('sb3/dqn-CartPole-v1',
                                           'dqn-CartPole-v1.zip', DQN)),
        ('sb3/ppo-CartPole-v1', load_agent('sb3/ppo-CartPole-v1',
                                           'ppo-CartPole-v1.zip', PPO)),
    ])
if "MountainCar" in experiment_data_path:
    agents_under_test.extend([
    ('sb3/dqn-MountainCar-v0', load_agent('sb3/dqn-MountainCar-v0',
                                          'dqn-MountainCar-v0.zip', DQN)),
    ('sb3/ppo-MountainCar-v0', load_agent('sb3/ppo-MountainCar-v0',
                                          'ppo-MountainCar-v0.zip', PPO)),
    ])
if "Acrobot" in experiment_data_path:
    agents_under_test.extend([
    ('sb3/dqn-Acrobot-v1', load_agent('sb3/dqn-Acrobot-v1',
                                          'dqn-Acrobot-v1.zip', DQN)),
    ('sb3/ppo-Acrobot-v1', load_agent('sb3/ppo-Acrobot-v1',
                                          'ppo-Acrobot-v1.zip', PPO)),
    ])


start = time.time()
clusters_of_interest = get_cluster_importance(agents_under_test[0][1], dim_red_pipeline, clustering_fun, importance_def=3)
end = time.time()
timing_info["cluster_importance"].append(end-start)

max_refinements = 20
num_learning_rounds, ep_per_round = 2, 100
for target in clusters_of_interest:

    output_file = open(f'pickles/diff_testing/diff_test_{env_name}_cluster_{target}.txt', 'a')
    sys.stdout = output_file

    refinements = 0
    while True:
        print(f'Retraining the model for {num_learning_rounds} learning rounds with {ep_per_round} episodes.')
        ir.iteratively_refine_model(num_learning_rounds, ep_per_round, goal_state=[target])
        refinements += num_learning_rounds
        model = ir.model
        print(f'Testing {",".join([x[0] for x in agents_under_test])} for {target}:')
        test_start = time.time()
        successfully_stopped, results = test_agents(env, env_name, model, agents_under_test, dim_red_pipeline,
                                                    clustering_fun, [target],
                                                    num_tests_per_agent=400)
        test_end = time.time()
        timing_info["testing"].append(test_end-test_start)
        if successfully_stopped:
            with open(f'pickles/diff_testing/diff_test_{env_name}_cluster_{target}.pickle', 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'pickles/diff_testing/diff_test_{env_name}_cluster_{target}_time.pickle', 'wb') as handle:
                pickle.dump(timing_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
                timing_info["testing"].clear()
                timing_info["mc"].clear()
                timing_info["refinement"].clear()
                timing_info["agent"].clear()

            output_file.close()
            break
        elif refinements >= max_refinements:
            print(f"Aborting {target} after {refinements} refinements")
            with open(f'pickles/diff_testing/diff_test_{env_name}_cluster_{target}_time.pickle', 'wb') as handle:
                pickle.dump(timing_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
                timing_info["testing"].clear()
                timing_info["mc"].clear()
                timing_info["refinement"].clear()
                timing_info["agent"].clear()
            output_file.close()
            break

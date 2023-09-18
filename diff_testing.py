import copy
from collections import Counter, defaultdict

import aalpy.paths
import gym
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
from stable_baselines3 import DQN, A2C, PPO

from agents import load_agent
from iterative_refinement import IterativeRefinement
from schedulers import PrismInterface, ProbabilisticScheduler, compute_weighted_clusters
from utils import load, mdp_from_state_setup, remove_nan, ACROBOT_GOAL, MOUNTAIN_CAR_GOAL, CARTPOLE_CUTOFF

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.7/bin/prism.bat"


# aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.8-linux64-x86/bin/prism"


def get_cluster_frequency(abstract_traces):
    cluster_counter = Counter()
    for at in abstract_traces:
        clusters = at[2::2]
        for c in clusters:
            cluster_counter[c] += 1

    return cluster_counter


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
                if ep_steps == CARTPOLE_CUTOFF:
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

        _, p = mannwhitneyu(agent_1_num_steps, agent_2_num_steps)
        if p > 0.05:
            print("Stopping early due to significant differance.")
            print(f'{agent_1} vs {agent_2}')
            print(f'p value: {p}')
            return False
        return True

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
                target_clusters, num_tests_per_agent=100, allowed_spurious_ration=0.2, verbose=True):
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

    while num_successful_tests < num_tests_per_agent:

        stop_early_based_on_significant_differance = stop_based_on_statistics(test_results_per_agent, env_name)
        if stop_early_based_on_significant_differance:
            break

        if total_test_num > 0 and total_test_num > 10 and spurious_tests_num / total_test_num > allowed_spurious_ration:
            # retrain the model
            print('Number of spurious test cases passed allowed successful/spurious ration.')
            print('Retrain the model')
            return False, test_results_per_agent

        for agent_name, agent in agents_under_test:
            # when true, execute tests with other agent
            switch_agent = False
            while True:

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
                        test_result = agent_suffix(agent, env, env_name, observation, ep_steps)

                        if env_name == 'LunarLander-v2' or env_name == 'CartPole-v1':
                            test_results_per_agent[agent_name][test_result] += 1
                        else:
                            agents_under_test[agent_name].append(test_result)

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
experiment_data_path = 'pickles/results/lda_mexp3_LunarLander-v2_num_traces_2500_lda_powerTransformer_n_clusters_1024_ri_25_ep_50.pk'

experiment_data = load(experiment_data_path)

last_iter_index = max(list(experiment_data.keys()))
abstract_traces = experiment_data[0]['learning_data']
model = mdp_from_state_setup(experiment_data[last_iter_index]['model'])
dim_red_pipeline = experiment_data[0]['dim_red_pipeline']
clustering_fun = experiment_data[0]['clustering_function']

cluster_counter = get_cluster_frequency(abstract_traces)

cluster_smallest_frequency = [x[0] for x in cluster_counter.most_common() if 'succ' not in x[0]]
cluster_smallest_frequency.reverse()

clusters_of_interest = cluster_smallest_frequency[0:15]

env_name = 'LunarLander-v2'
env = gym.make(env_name, )

ir = IterativeRefinement(env=env, env_name=env_name, abstract_traces=abstract_traces,
                         initial_model=model,
                         dim_reduction_pipeline=dim_red_pipeline,
                         clustering_fun=clustering_fun,
                         experiment_name_prefix='test_diff')

ir.current_iteration = last_iter_index + 1
ir.results = experiment_data

# model = ir.model

agents_under_test = [
    # ('araffin/dqn-LunarLander-v2', load_agent('araffin/dqn-LunarLander-v2',
    #                                                            'dqn-LunarLander-v2.zip', DQN)),
    # ('araffin/a2c-LunarLander-v2', load_agent('araffin/a2c-LunarLander-v2',
    #                                           'a2c-LunarLander-v2.zip', A2C)),
    # ('sb3/dqn-LunarLander-v2', load_agent('sb3/dqn-LunarLander-v2',
    #                                       'dqn-LunarLander-v2.zip', DQN)),
    ('sb3/a2c-LunarLander-v2', load_agent('sb3/a2c-LunarLander-v2',
                                          'a2c-LunarLander-v2.zip', A2C)),
    ('sb3/ppo-LunarLander-v2', load_agent('sb3/ppo-LunarLander-v2',
                                          'ppo-LunarLander-v2.zip', PPO)),
]

num_learning_rounds, ep_per_round = 2, 50
for target in clusters_of_interest:
    while True:
        print(f'Retraining the model for {num_learning_rounds} learning rounds with {ep_per_round} episodes.')
        ir.iteratively_refine_model(num_learning_rounds, ep_per_round, goal_state=[target])
        model = ir.model
        print(f'Testing {",".join([x[0] for x in agents_under_test])} for {target}:')
        successfully_stopped, results = test_agents(env, env_name, model, agents_under_test, dim_red_pipeline,
                                                    clustering_fun, [target],
                                                    num_tests_per_agent=200)

        if successfully_stopped:
            break
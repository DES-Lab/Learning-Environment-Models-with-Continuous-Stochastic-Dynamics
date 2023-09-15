import copy
from collections import Counter, defaultdict

import aalpy.paths
import gym
import numpy as np
from scipy.stats import fisher_exact
from stable_baselines3 import DQN, A2C, PPO

from agents import load_agent
from iterative_refinement import IterativeRefinement
from schedulers import PrismInterface, ProbabilisticScheduler, compute_weighted_clusters
from utils import load, mdp_from_state_setup, remove_nan

# aalpy.paths.path_to_prism = "C:/Program Files/prism-4.7/bin/prism.bat"
aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.8-linux64-x86/bin/prism"


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


def agent_suffix(agent, env, env_name, last_observation):
    observation = last_observation
    while True:
        action, _ = agent.predict(observation)
        observation, reward, done, _ = env.step(action)

        if done:
            if env_name == 'LunarLander-v2':
                if reward >= 80:
                    return 'Landed'
                if reward < -80:
                    return 'Crash'
                else:
                    return 'Time_out'


def test_agents(env, env_name, model, agents_under_test, dim_reduction_pipeline, clustering_fun,
                target_clusters, num_tests_per_agent=100, verbose=True):

    assert len(agents_under_test) == 2
    agent_test_results = defaultdict(Counter)

    # due to a bug in alergia
    remove_nan(model)
    # Make model input complete
    model.make_input_complete('sink_state')

    stop_testing = False

    num_successful_tests = 0

    scheduler = PrismInterface(target_clusters, model).scheduler
    if scheduler is None:
        print('Scheduler could not be computed.')
        assert False

    scheduler = ProbabilisticScheduler(scheduler, truly_probabilistic=True)

    while num_successful_tests < num_tests_per_agent:

        agent_1, agent_2 = agents_under_test[0][0], agents_under_test[1][0]
        f1 = agent_test_results[agent_1]['Crash'] + agent_test_results[agent_1]['Time_out']
        f2 = agent_test_results[agent_2]['Crash'] + agent_test_results[agent_2]['Time_out']

        n1 = agent_test_results[agent_1]['Landed'] + f1
        n2 = agent_test_results[agent_2]['Landed'] + f2

        diff, ratio = diff_test(f1, n1, f2, n2, 0.05)
        if n1 > 0 and n2 > 0 and diff:
            print("Stopping early due to significant differance.")
            print(f'{agent_1} vs {agent_2}')
            print(f"{f1 / n1} vs {f2 / n2}")
            break
        for agent_name, agent in agents_under_test:
            had_success = False
            while True:
                scheduler.reset()
                env.reset()
                ep_rew = 0

                while True:

                    scheduler_input = scheduler.get_input()
                    if scheduler_input is None:
                        print('Could not schedule an action.')
                        break

                    action = np.array(int(scheduler_input[1:]))

                    observation, reward, done, _ = env.step(action)

                    ep_rew += reward

                    if dim_reduction_pipeline is not None:
                        abstract_obs = dim_reduction_pipeline.transform(np.array([observation]))
                    else:
                        abstract_obs = [observation.reshape(1, -1)]

                    reached_cluster = np.argmin(clustering_fun.transform(abstract_obs)) #clustering_fun.predict(abstract_obs)[0]
                    reached_cluster = f'c{reached_cluster}'

                    if reached_cluster in target_clusters:
                        if verbose:
                            print('Target cluster reached.')
                            print('Switching to an agent under test.')
                        test_result = agent_suffix(agent, env, env_name, observation)
                        agent_test_results[agent_name][test_result] += 1

                        num_successful_tests += 1
                        had_success = True
                        break

                    weighted_clusters = compute_weighted_clusters(scheduler, abstract_obs, scheduler_input,
                                                                  clustering_fun,
                                                                  len(set(clustering_fun.labels_)))

                    step_successful = scheduler.step_to(scheduler_input, weighted_clusters)

                    if not step_successful or done:
                        break
                if had_success:
                    break

    for agent_name, test_res in agent_test_results.items():
        print(agent_name)
        for k, v in test_res.items():
            print(f'{k} : {v}')
        print('---------------')


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

for target in clusters_of_interest:
    print(f'Testing Agents for {target}')
    ir.iteratively_refine_model(2, 50, goal_state=[target])
    model = ir.model
    test_agents(env, env_name, model, agents_under_test, dim_red_pipeline, clustering_fun, [target],\
                num_tests_per_agent=200)

import pickle
from collections import defaultdict, Counter
from itertools import product, cycle
from random import choice
from statistics import mean

import numpy as np
from tqdm import tqdm

# pip install ufal.pybox2d
import sys

sys.path.append('.')

from robustness_smc.agents_under_test import *


def smc(agent, env, num_policy_moves, num_random_moves, agent_name, available_actions, mutation_type, evaluate_obs,
        num_runs=300, render=False):
    assert mutation_type in {'action', 'observation'}

    episode_rewards = []
    episode_lens = []

    results_dict = Counter()

    # create a cycle of policy-random actions
    action_sequence = ['policy' for _ in range(num_policy_moves)]
    action_sequence.extend(['random' for _ in range(num_random_moves)])

    for _ in tqdm(range(num_runs)):
        # used for random continuous actions
        previous_actions = []
        # reset the cycle
        action_cycle = cycle(action_sequence)

        episode_reward = 0
        episode_step_counter = 0
        obs = env.reset()
        while True:
            episode_step_counter += 1

            # mutate actions
            if mutation_type == 'action':
                action, _ = agent.predict(obs)

                if available_actions == 'continuous':
                    previous_actions.append(action)

                if next(action_cycle) == 'random':
                    if available_actions != 'continuous':
                        non_optimal_actions = available_actions.copy()
                        non_optimal_actions.remove(action)
                        action = choice(non_optimal_actions)
                    else:
                        # randomly sample any of previous actions, but not one of the last 5 actions as those ones might
                        # be too close to current optimal one?
                        action = choice(previous_actions[:-5])
            # mutate observation
            else:
                if next(action_cycle) == 'random':
                    noise = np.random.normal(0, 0.5, obs.shape)
                    obs += noise
                action, _ = agent.predict(obs)

            obs, reward, done, info = env.step(action)
            episode_reward += reward

            if render:
                env.render()

            if done:
                episode_rewards.append(episode_reward)
                info['ep_len'] = episode_step_counter
                episode_lens.append(episode_step_counter)
                evaluate_obs(obs, reward, done, info, results_dict)

                break

    goal_reached = round(results_dict['goal'] / num_runs * 100, 2)
    crash = round(results_dict['crash'] / num_runs * 100, 2)
    time_out = round(results_dict['timeout'] / num_runs * 100, 2)
    episode_len = round(mean(episode_lens), 2)
    episode_rewards = round(mean(episode_rewards), 2)

    print('--------------------------------------------------------------------------------------')
    print(f'Agent: {agent_name}')
    print(f'SMC Statistics for {num_policy_moves, num_random_moves} configuration.')
    print(f'Goal       : {goal_reached}')
    print(f'Crash      : {crash}')
    print(f'Time-out   : {time_out}')
    print(f'Episode len: {episode_len}')

    print(f'Mean Reward: {episode_rewards}')

    return goal_reached, crash, time_out, episode_rewards, episode_len


def run_all_experiments():
    experiments = [['lunar_lander', get_lunar_lander_agents_smc],
                   ['cartpole', get_cartpole_agents_smc],
                   ['mountain_car', get_mountaincar_agents_smc],
                   ['bipedal_walker', get_bipedal_walker_agents],
                   ['walker2d', get_walker2d_agents_smc]]

    experiments_with_mutator = []
    for exp in experiments:
        for m in ['action', 'observation']:
            e = exp.copy()
            e.insert(1, m)
            experiments_with_mutator.append(e)

    num_policy_steps = list(range(10, 60, 10))
    num_random_steps = list(range(1, 11, 2))

    experiment_configs = list(product(num_policy_steps, num_random_steps))

    num_runs_per_exp = 150

    for exp_name, mutation_type, exp_getter_function, in experiments_with_mutator:
        agent_configs, available_actions, env, evaluate_obs_fun = exp_getter_function()
        experiment_results = defaultdict(dict)

        for num_policy_moves, num_random_moves in experiment_configs:
            for agent_name, agent in agent_configs:
                print(exp_name)
                data = smc(agent, env,
                           num_policy_moves=num_policy_moves, num_random_moves=num_random_moves,
                           available_actions=available_actions, evaluate_obs=evaluate_obs_fun,
                           mutation_type=mutation_type, agent_name=agent_name, num_runs=num_runs_per_exp)
                experiment_results[agent_name][(num_policy_moves, num_random_moves)] = data

        with open(f'smc_{exp_name}_{mutation_type}.pickle', 'wb') as handle:
            pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    run_all_experiments()
    exit()
    num_policy_steps = list(range(10, 60, 10))
    num_random_steps = list(range(1, 11, 2))

    experiment_configs = list(product(num_policy_steps, num_random_steps))

    exp_name = 'lunar_lander'
    mutation_type = 'observation'  # 'action' or 'observation'
    agent_configs, available_actions, env, evaluate_obs_fun = get_lunar_lander_agents_smc()

    experiment_results = defaultdict(dict)

    for num_policy_moves, num_random_moves in experiment_configs:
        for agent_name, agent in agent_configs:
            data = smc(agent, env,
                       num_policy_moves=num_policy_moves, num_random_moves=num_random_moves,
                       available_actions=available_actions, evaluate_obs=evaluate_obs_fun,
                       mutation_type=mutation_type, agent_name=agent_name, num_runs=120)
            experiment_results[agent_name][(num_policy_moves, num_random_moves)] = data

    with open(f'smc_{exp_name}_{mutation_type}.pickle', 'wb') as handle:
        pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

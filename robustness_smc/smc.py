import pickle
from collections import defaultdict, Counter
from itertools import product, cycle
from random import choice
from statistics import mean

from tqdm import tqdm

# pip install ufal.pybox2d
from robustness_smc.agents_under_test import *


def smc(agent, env, num_policy_moves, num_random_moves, agent_name, available_actions, evaluate_obs, num_runs=300, render=False):
    episode_rewards = []
    results_dict = Counter()

    action_sequance = ['policy' for _ in range(num_policy_moves)]
    action_sequance.extend(['random' for _ in range(num_random_moves)])

    for _ in tqdm(range(num_runs)):
        obs = env.reset()

        episode_reward = 0

        previous_actions = []

        action_cycle = cycle(action_sequance)

        episode_step_counter = 0
        while True:
            episode_step_counter = 0
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

            obs, reward, done, info = env.step(action)
            episode_reward += reward

            if render:
                env.render()

            if done:
                episode_rewards.append(episode_reward)
                info['ep_len'] = episode_step_counter
                evaluate_obs(obs, reward, done, info, results_dict)

                break

    goal_reached = round(results_dict['goal'] / num_runs * 100, 2)
    crash = round(results_dict['crash'] / num_runs * 100, 2)
    time_out = round(results_dict['timeout'] / num_runs * 100, 2)

    print('--------------------------------------------------------------------------------------')
    print(f'Agent: {agent_name}')
    print(f'SMC Statistics for {num_policy_moves, num_random_moves} configuration.')
    print(f'Goal       : {goal_reached}')
    print(f'Crash      : {crash}')
    print(f'Time-out   : {time_out}')

    print(f'Mean Reward: {mean(episode_rewards)}')

    return goal_reached, crash, time_out, mean(episode_rewards)


if __name__ == '__main__':

    num_policy_steps = list(range(10, 60, 10))
    num_random_steps = list(range(1, 11, 2))

    experiment_configs = list(product(num_policy_steps, num_random_steps))

    # experiment_configs = [(10, 1), (20, 1),] #  (30, 1), (50, 5), (50, 10)
    agent_configs, available_actions, env, evaluate_obs_fun = get_bipedal_walker_agents()

    experiment_results = defaultdict(dict)

    for num_policy_moves, num_random_moves in experiment_configs:
        for agent_name, agent in agent_configs:
            data = smc(agent, env,
                       num_policy_moves=num_policy_moves, num_random_moves=num_random_moves,
                       available_actions=available_actions, evaluate_obs=evaluate_obs_fun,
                       agent_name=agent_name, num_runs=10)
            experiment_results[agent_name][(num_policy_moves, num_random_moves)] = data

    with open('smc_bipedal_walker.pickle', 'wb') as handle:
        pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

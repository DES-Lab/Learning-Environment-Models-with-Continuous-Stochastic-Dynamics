import pickle
from collections import defaultdict
from itertools import product
from random import choice
from statistics import mean

from tqdm import tqdm

# pip install ufal.pybox2d
from agents_under_test import get_lunar_lander_agents_smc, get_cartpole_agents_smc, get_mountaincar_agents_smc, \
    get_bipedal_walker_agents


def smc(agent, env, num_optimal_moves, num_random_moves, agent_name, available_actions, num_runs=300, render=False):
    goal_reached, crash, time_out, rewards = 0, 0, 0, []

    for _ in tqdm(range(num_runs)):
        obs = env.reset()

        optimal_counter = 0
        random_counter = 0
        episode_reward = 0

        previous_actions = []
        while True:
            action, _ = agent.predict(obs)

            if available_actions == 'bipedal_walker':
                previous_actions.append(action)

            if optimal_counter < num_optimal_moves:
                optimal_counter += 1
            else:
                random_counter += 1
                if random_counter == num_random_moves:
                    optimal_counter = 0
                    random_counter = 0

                if available_actions != 'bipedal_walker':
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
                rewards.append(episode_reward)
                if reward == 100:
                    goal_reached += 1
                elif reward == -100:
                    crash += 1
                else:
                    time_out += 1

                break

    goal_reached = round(goal_reached / num_runs * 100, 2)
    crash = round(crash / num_runs * 100, 2)
    time_out = round(time_out / num_runs * 100, 2)
    print('--------------------------------------------------------------------------------------')
    print(f'Agent: {agent_name}')
    print(f'SMC Statistics for {num_optimal_moves, num_random_moves} configuration.')
    print(f'Goal       : {goal_reached}')
    print(f'Crash      : {crash}')
    print(f'Time-out   : {time_out}')

    print(f'Mean Reward: {mean(rewards)}')

    return goal_reached, crash, time_out, mean(rewards)


num_policy_steps = list(range(10, 60, 10))
num_random_steps = list(range(1, 11, 2))

experiment_configs = list(product(num_policy_steps, num_random_steps))

# experiment_configs = [(10, 1), (20, 1),] #  (30, 1), (50, 5), (50, 10)
agent_configs, available_actions, env = get_bipedal_walker_agents()

experiment_results = defaultdict(dict)

for num_policy_moves, num_random_moves in experiment_configs:
    for agent_name, agent in agent_configs:
        data = smc(agent, env, num_optimal_moves=num_policy_moves, num_random_moves=num_random_moves,
                   available_actions=available_actions, agent_name=agent_name, num_runs=200)
        experiment_results[agent_name][(num_policy_moves, num_random_moves)] = data

with open('smc_bipedal_walker.pickle', 'wb') as handle:
    pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

import gym
from random import choice
from statistics import mean
from tqdm import tqdm

from stable_baselines3 import DQN, A2C

from agents import load_agent, get_lunar_lander_agents_smc, get_mountaincar_agents_smc


def smc(agent, env, num_optimal_moves, num_random_moves, agent_name, available_actions, num_runs=100, render=False):
    goal_reached, crash, time_out, rewards = 0, 0, 0, []

    for _ in tqdm(range(num_runs)):
        obs = env.reset()

        optimal_counter = 0
        random_counter = 0
        episode_reward = 0
        while True:
            action, _ = agent.predict(obs)

            if optimal_counter < num_optimal_moves:
                optimal_counter += 1
            else:
                random_counter += 1
                if random_counter == num_random_moves:
                    optimal_counter = 0
                    random_counter = 0

                non_optimal_actions = available_actions.copy()
                non_optimal_actions.remove(action)
                action = choice(non_optimal_actions)

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

    print('--------------------------------------------------------------------------------------')
    print(f'Agent: {agent_name}')
    print(f'SMC Statistics for {num_optimal_moves, num_random_moves} configuration.')
    print(f'Goal       : {round(goal_reached / num_runs * 100, 2)}')
    print(f'Crash      : {round(crash / num_runs * 100, 2)}')
    print(f'Time-out   : {round(time_out / num_runs * 100, 2)}')

    print(f'Mean Reward: {mean(rewards)}')

    return goal_reached, crash, mean(rewards)


experiment_configs = [(10, 1), (20, 1), (30, 3), (50, 5), (50, 10)]
agent_configs, available_actions, env = get_mountaincar_agents_smc()
for num_policy_moves, num_random_moves in experiment_configs:
    for agent_name, agent in agent_configs:
        smc(agent, env, num_optimal_moves=num_policy_moves, num_random_moves=num_random_moves,
            available_actions = available_actions, agent_name=agent_name, render=False)

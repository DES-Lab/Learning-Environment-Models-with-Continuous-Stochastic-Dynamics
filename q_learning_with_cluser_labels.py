import random

import gym
import numpy as np

from utils import load

env = gym.make('LunarLander-v2')

clustering_function = load('pickle_files/k_means_scale_True_64_6000.pickle')

q_table = np.zeros([64, env.action_space.n])

# Hyper parameters
alpha = 0.1
gamma = 0.6
epsilon = 0.2

num_training_episodes = 10000

# For plotting metrics
all_epochs = []

training_terminated = 0
training_reward_sum = 0

for i in range(1, num_training_episodes + 1):
    state = env.reset()
    obs = clustering_function.predict(state.reshape(1,-1))
    reward, terminated = 0, False
    done = False
    while not done:

        action = env.action_space.sample() if random.random() < epsilon else np.argmax(q_table[obs])
        # env.render()
        next_state, reward, done, info = env.step(action)
        new_obs = clustering_function.predict(next_state.reshape(1, -1))

        old_value = q_table[obs, action]

        next_max = np.max(q_table[new_obs])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[obs, action] = new_value

        reward += reward

        if done:
            training_reward_sum += reward

        obs = new_obs

    if i % 100 == 0:
        print(f"Episode: {i}, Reward: {reward}, Terminated: {terminated}")

print("Training finished.")
print(f"Total reward: {training_reward_sum}")
print(f"Total death state reached: {training_terminated}")

total_epochs = 0
episodes = 100

goals_reached = 0
for _ in range(episodes):
    state = env.reset()
    obs = clustering_function.predict(state.reshape(1,-1))

    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[obs])
        state, reward, done, info = env.step(action)
        obs = clustering_function.predict(state.reshape(1, -1))

        if reward == 100 and done:
            goals_reached += 1

        epochs += 1

    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Total Number of Goal reached: {goals_reached}")
print(f"Average timesteps per episode: {total_epochs / episodes}")
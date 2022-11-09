import random
import torch
import gym
from aalpy.learning_algs import run_Alergia, run_JAlergia
from aalpy.utils import load_automaton_from_file
from stable_baselines3 import DQN
from stable_baselines3.common.utils import obs_as_tensor

from agents import load_agent
from prism_scheduler import PrismInterface
from utils import save_samples_to_file, delete_file

dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
env = gym.make("LunarLander-v2")

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}

import aalpy.paths

aalpy.paths.path_to_prism = 'C:/Program Files/prism-4.6/bin/prism.bat'


def get_q_value(obs, model):
    observation = obs.reshape((-1,) + model.observation_space.shape)
    # observation = obs_as_tensor(observation, 'cpu')
    observation = torch.tensor(observation)
    return int(max(model.q_net(observation)[0].tolist()))


alergia_samples = []
with torch.no_grad():
    for _ in range(10000):
        obs = env.reset()
        sample = ['INIT']
        while True:
            action, state = dqn_agent.predict(obs)

            obs, reward, done, _ = env.step(action)
            q_value = get_q_value(obs, dqn_agent)
            if q_value < 0:
                q_value = 'NEGATIVE'
            output = f'c{q_value}'
            if done and reward == 100:
                output = 'GOAL'
            sample.append((action_map[int(action)], output))
            if done:
                break
        alergia_samples.append(sample)

jalergia_samples = 'rewardSample.txt'
save_samples_to_file(alergia_samples, jalergia_samples)

model = run_JAlergia('rewardSample.txt', 'mdp', 'alergia.jar', heap_memory='-Xmx4G', optimize_for='memory')
model.save('reward_automaton')
model = load_automaton_from_file('reward_automaton.dot', 'mdp')
model.make_input_complete('sink_state')

delete_file(jalergia_samples)

prism_interface = PrismInterface('GOAL', model)

for _ in range(1000):
    obs = env.reset()
    prism_interface.reset()

    reward = 0
    while True:
        action = prism_interface.get_input()
        if action is None:
            print('Cannot schedule an action')
            break

        obs, rew, done, info = env.step(input_map[action])
        reward += rew

        q_value = get_q_value(obs, dqn_agent)
        if q_value < 0:
            q_value = 'NEGATIVE'
        output = f'c{q_value}'
        if done and reward == 100:
            output = 'GOAL'
        reached_state = prism_interface.step_to(action, output)
        env.render()
        if not reached_state:
            possible_outs = prism_interface.poss_step_to(action)
            prism_interface.step_to(action, random.choice(possible_outs))
        if done:
            print(env.game_over)
            if not env.game_over:
                print(rew)
                # import time
                # time.sleep(2)
            print('Episode reward: ', reward)
            if reward > 1:
                print('Success', reward)
            break

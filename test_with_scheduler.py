import gym
import aalpy.paths

from prism_scheduler import PrismInterface
from aalpy.utils import load_automaton_from_file

from utils import load

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.6/bin/prism.bat"

model = load_automaton_from_file('mdp_dqn.dot', 'mdp')
model.make_input_complete(missing_transition_go_to='sink_state')
prism_interface = PrismInterface("DONE", model)

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v:k for k, v in action_map.items()}

clustering_function = load('k_means_16.pickle')

env = gym.make('LunarLander-v2')

for _ in range(10):
    obs = env.reset()
    prism_interface.reset()
    reward = 0
    while True:
        action = prism_interface.get_input()
        action = input_map[action]

        obs, rew, done, info = env.step(action)
        reward += rew

        obs = clustering_function.predict(obs.reshape(1,-1))
        reached_state = prism_interface.step_to(action, obs)
        env.render()
        if reached_state is None:
            done = True
            reward = -1000
            print('Run into state that is unreachable in the model.')
        if done:
            print('Episode reward: ', reward)
            break



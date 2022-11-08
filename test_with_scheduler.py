import gym
import aalpy.paths

from prism_scheduler import PrismInterface
from aalpy.utils import load_automaton_from_file
from sklearn.metrics import euclidean_distances

from utils import load


aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.7-linux64/bin/prism"

model = load_automaton_from_file('mdp_combined_scale_True_48_10000.dot', 'mdp')
model.make_input_complete(missing_transition_go_to='sink_state')
prism_interface = PrismInterface("succ", model)

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v:k for k, v in action_map.items()}

clustering_function = load('k_means_scale_True_48_10000.pickle')
scaler = load("standard_scaler_10000.pickle")

env = gym.make('LunarLander-v2')
scale = True

def take_best_out(prism_interface, scaler, clustering, concrete_obs, action,
                  possible_outs,scale):
    first_out = possible_outs[0]
    min_dist = 1e30
    min_l = None

    for o in possible_outs:
        for i,corr_center in enumerate(clustering.cluster_centers_):
            cluster = clustering_function.predict(corr_center.reshape(1, -1))[0]
            if f"c{cluster}" in o:
                distance = euclidean_distances(concrete_obs, corr_center.reshape(1, -1))
                if min_dist is None or distance < min_dist:
                    min_dist = distance
                    min_l = o
    prism_interface.step_to(action,min_l)


for _ in range(1000):
    obs = env.reset()
    conc_obs = obs.reshape(1, -1)

    if scale:
        conc_obs = scaler.transform(conc_obs)
    obs = f'c{clustering_function.predict(conc_obs)[0]}'

    prism_interface.reset()
    # prism_interface.step_to('right_engine', obs)
    reward = 0
    while True:
        action = prism_interface.get_input()
        if action is None:
            print('Cannot schedule an action')
            break
        concrete_action = input_map[action]

        obs, rew, done, info = env.step(concrete_action)
        reward += rew
        conc_obs = obs.reshape(1,-1)

        if scale:
            conc_obs = scaler.transform(conc_obs)
        obs = f'c{clustering_function.predict(conc_obs)[0]}'
        reached_state = prism_interface.step_to(action, obs)
        env.render()
        if not reached_state:
            # done = True
            # reward = -1000
            #print('Run into state that is unreachable in the model.')
            possible_outs = prism_interface.poss_step_to(action)
            take_best_out(prism_interface,scaler,clustering_function,conc_obs,action,
                          possible_outs,scale)
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



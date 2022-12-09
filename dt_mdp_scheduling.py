
import aalpy.paths
import gym
from aalpy.utils import load_automaton_from_file

from dt_mdp_learning import build_tree_copy
from prism_scheduler import PrismInterface, ProbabilisticScheduler
from utils import load, save


def compute_weighted_outs(scheduler, dt, tree_copy, obs, action,possible_outs):
    predicted_id = int(obs[1:])
    weighted_outputs = dict()
    for o in possible_outs:
        labels = o.split("__")
        for label in labels:
            if label.startswith("c"):
                # print("FOO")
                o_leave_id = int(o[1:])
                path_to_root_predicted = tree_copy.paths_to_root[predicted_id]
                path_to_root_o = tree_copy.paths_to_root[o_leave_id]
                common_path_len = 0
                # print(path_to_root_predicted)
                # print(path_to_root_o)
                for i in range(min(len(path_to_root_predicted), len(path_to_root_o))-1):
                    if path_to_root_predicted[-i-1] == path_to_root_o[-i-1]:
                        common_path_len += 1
                    else:
                        break
                weighted_outputs[label] = common_path_len
    weight_sum = sum(weighted_outputs.values())
    for label in weighted_outputs:
        weighted_outputs[label] = weighted_outputs[label] / weight_sum
    return weighted_outputs
def take_best_out(scheduler, dt, tree_copy, obs, action,possible_outs):
    longest_common_path = 0
    best_label = None
    predicted_id = int(obs[1:])
    for o in possible_outs:
        labels = o.split("__")
        for label in labels:
            if label.startswith("c"):
                # print("FOO")
                o_leave_id = int(o[1:])
                path_to_root_predicted = tree_copy.paths_to_root[predicted_id]
                path_to_root_o = tree_copy.paths_to_root[o_leave_id]
                common_path_len = 0
                # print(path_to_root_predicted)
                # print(path_to_root_o)
                for i in range(min(len(path_to_root_predicted), len(path_to_root_o))-1):
                    if path_to_root_predicted[-i-1] == path_to_root_o[-i-1]:
                        common_path_len += 1
                    else:
                        break
                if common_path_len > longest_common_path:
                    longest_common_path = common_path_len
                    best_label = label
    print(best_label)
    scheduler.step_to(action, best_label)


def run_episode(env,scheduler,dt, tree_copy):
    obs = env.reset()
    conc_obs = obs.reshape(1, -1)

    obs = f'c{dt.predict(conc_obs)[0]}'

    scheduler.reset()
    # prism_interface.step_to('right_engine', obs)
    reward = 0
    while True:
        action = scheduler.get_input()
        if action is None:
            print('Cannot schedule an action')
            break
        concrete_action = input_map[action]

        obs, rew, done, info = env.step(concrete_action)
        reward += rew
        conc_obs = obs.reshape(1, -1)

        obs = f'c{dt.apply(conc_obs)[0]}'
        possible_outs = scheduler.poss_step_to(action)
        weighted_obs = compute_weighted_outs(scheduler,dt,tree_copy,obs,action,possible_outs)
        reached_state = scheduler.step_to(action, weighted_obs)
        env.render()
        if not reached_state:
            done = True
            reward = -1000
            print('Run into state that is unreachable in the model.')
            # possible_outs = scheduler.poss_step_to(action)
            # take_best_out(scheduler, dt, tree_copy, obs, action,
            #               possible_outs)
        if done:
            print(env.game_over)
            if not env.game_over:
                print(rew)
            print('Episode reward: ', reward)
            if reward > 1:
                print('Success', reward)
            break

num_traces_dt = 8000
max_leaves = 512
num_traces = 2300
scale = True


environment = 'LunarLander-v2'
aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.7-linux64/bin/prism"
agent_steps = 0

model = load_automaton_from_file(f'dt_mdp_LunarLander-v2_{num_traces_dt}_{max_leaves}_{num_traces}.dot', 'mdp')
model.make_input_complete(missing_transition_go_to='sink_state')
sched_file_name = f"dt_prism_interface_{num_traces_dt}_{max_leaves}_{num_traces}"
# prism_interface = load(sched_file_name)
# if prism_interface is None:
prism_interface = PrismInterface(["succ"], model)
    #save(prism_interface,sched_file_name)

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}

dt = load(f'dt_{environment}_{num_traces_dt}_{max_leaves}')

env = gym.make(environment)
tree_copy_file_name = f"tree_copy_{num_traces_dt}_{max_leaves}_{num_traces}"
tree_copy = load(tree_copy_file_name)
if tree_copy is None:
    tree_copy = build_tree_copy(dt.tree_)
    tree_copy.compute_aux_information()
    save(tree_copy,tree_copy_file_name)

prob_scheduler = ProbabilisticScheduler(prism_interface.scheduler,True, max_state_size=128)
for i in range(200):
    run_episode(env,prob_scheduler,dt, tree_copy)
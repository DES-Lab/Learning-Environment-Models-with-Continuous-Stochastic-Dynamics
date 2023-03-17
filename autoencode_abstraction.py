import math
import os
import pickle
import random
from collections import defaultdict

from statistics import mean

import gym
import torch
import aalpy.paths
from aalpy.learning_algs import run_Alergia, run_JAlergia
from aalpy.utils import load_automaton_from_file
from stable_baselines3 import DQN
from torch.utils.data import DataLoader
from tqdm import tqdm

from agents import load_agent
from prism_scheduler import PrismInterface, ProbabilisticScheduler
from utils import load, get_traces_from_policy, save

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}


def remove_nan(mdp):
    changed = False
    for s in mdp.states:
        to_remove = []
        for input in s.transitions.keys():
            is_nan = False
            for t in s.transitions[input]:
                if math.isnan(t[1]):
                    is_nan = True
                    to_remove.append(input)
                    break
            if not is_nan:
                if abs(sum(map(lambda t: t[1], s.transitions[input])) - 1) > 1e-6:
                    to_remove.append(input)
        for input in to_remove:
            changed = True
            s.transitions.pop(input)
    return changed


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(8, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8),
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        out = self.encoder(x).item()
        rounded = round(out, 1)
        out_str = str(rounded).replace('.', '_')
        if out < 0:
            out_str = str(abs(rounded)).replace('.', '_')
            return f'sn{out_str}'
        return f's{out_str}'

    def get_approximation(self, x):
        if 'sn' in x:
            return -float(x[2:].replace('_', '.'))
        return float(x[1:].replace('_', '.'))


def train(model, data, epochs, optimizer, loss_function):
    for epoch in range(epochs):

        losses = []

        for obs in data:
            reconstructed = model(obs)

            # Calculating the loss function
            loss = loss_function(obs, reconstructed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            losses.append(loss.item())

        print(f'Epoch {epoch}, loss : {mean(losses)}')


def display_examples(model, data):
    for obs in data:
        reconstructed = model(obs)
        print('---------------------')
        print(obs.detach().numpy()[0].tolist())
        print(reconstructed.detach().numpy()[0].tolist())
        print('Latent space', model.encoder(obs))


def create_alergia_sequances(model, traces):
    alergia_sequances, goal_states = [], []
    print('Creating Alergia sequances')
    for trace in tqdm(traces):
        seq = ['INIT']
        for obs, action, reward, done in trace:
            abstract_obs = model.encode(torch.tensor(obs))
            if done:
                if reward == 100:
                    goal_states.append(abstract_obs)
                if reward == -100:
                    abstract_obs = 'CRASH'
            seq.extend((action_map[int(action)], abstract_obs))
        alergia_sequances.append(seq)

    return alergia_sequances, goal_states


def visualize_abstraction(model, traces):
    cluster_coordinate_map = defaultdict(list)
    print('Creating visualization data')
    for trace in tqdm(traces):
        for obs, action, reward, done in trace[1:]:
            abstract_obs = model.encode(torch.tensor(obs))
            cluster_coordinate_map[abstract_obs].append((obs[0][0], obs[0][1]))

    import matplotlib.pyplot as plt

    for label, coordinate_tuples in cluster_coordinate_map.items():
        x, y = [i[0] for i in coordinate_tuples], [i[1] for i in coordinate_tuples]
        plt.scatter(x, y, label=label)

    plt.show()


def test_with_scheduler(env, model, alergia_model, goal_states, num_episodes=100):
    prism_interface = PrismInterface(goal_states, alergia_model)

    scheduler = prism_interface.scheduler

    total_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        scheduler.reset()
        alergia_model.reset_to_initial()
        # hack to reach first state in the model

        ep_rew = 0

        while True:
            action = scheduler.get_input()
            print(action)
            assert action is not None
            obs, rew, done, _ = env.step(input_map[action])
            env.render()

            abstract_obs = model.encode(torch.tensor(obs))

            if done and rew == -100:
                abstract_obs = 'CRASH'

            reached_state = scheduler.step_to(action, abstract_obs)

            if not reached_state:
                encoder_output = model.get_approximation(abstract_obs)

                min_distance, selected_node = None, None

                for transitions in alergia_model.current_state.transitions.values():
                    for node, _ in transitions:
                        if not min_distance:
                            min_distance = abs(model.get_approximation(node.output) - encoder_output)
                            selected_node = node
                        else:
                            if 'sink_state' in node.output or 'CRASH' in node.output or 'GOAL' in node.output:
                                continue
                            node_distance = abs(model.get_approximation(node.output) - encoder_output)
                            if min_distance > node_distance:
                                selected_node = node
                                min_distance = node_distance

                assert selected_node

                print(action, selected_node.output)
                reached_state = scheduler.step_to(action, selected_node.output)
                assert reached_state

            alergia_model.step_to(action, reached_state)

            ep_rew += rew

            if done:
                print("Episode reward", ep_rew)
                total_rewards.append(ep_rew)
                break


#################################################
def compute_history_based_prediction(alergia_traces, max_history_len=5):
    action_dict = defaultdict(dict)
    for trace in alergia_traces:
        # TODO
        trace_splits = [trace[j:j + i] for i in range(1, max_history_len + 1) for j in range(1, len(trace))]
        # print(trace_splits)
        for split in trace_splits:
            clusters = tuple(c[1] for c in split)
            action = split[-1][0]
            if action not in action_dict[clusters].keys():
                action_dict[clusters][action] = 0
            action_dict[clusters][action] += 1

    return action_dict


def choose_action(obs, action_dict, based_on='confidence'):
    all_splits = [tuple(obs[len(obs) - i - 1:]) for i in range(len(obs))]
    all_splits.reverse()
    chosen_action = None
    if based_on == 'longest':
        for split in all_splits:
            if split in action_dict.keys():
                chosen_action = max(action_dict[split], key=action_dict[split].get)
                break
    if based_on == 'probabilistic_longest':
        for split in all_splits:
            if split in action_dict.keys():
                actions = list(action_dict[split].keys())
                weights = list(action_dict[split].values())
                chosen_action = random.choices(actions, weights=weights)[0]
                break

    elif based_on == 'confidence':
        split_action_lists = []  # compute frequqnct
        for split in all_splits:
            pass
    elif based_on == 'majority_vote':
        pass

    return chosen_action


def evaluate(env, action_dict, model,
             history_window_size=5,
             num_episodes=100,
             strategy='longest',
             render=True):
    all_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        history = []
        ep_rew = 0
        while True:
            history.append(model.encode(torch.tensor(obs)))
            history = history[-history_window_size:]

            abstract_action = choose_action(tuple(history), action_dict, based_on=strategy)
            if not abstract_action:
                break
            action = input_map[abstract_action]
            obs, rew, done, info = env.step(action)
            if render:
                env.render()
            ep_rew += rew
            if done:
                print('Episode reward', ep_rew)
                all_rewards.append(ep_rew)
                break
    print('Mean reward:', mean(all_rewards))


#################################################

env_name = "LunarLander-v2"

num_traces = 10
aalpy.paths.path_to_prism = "C:/Program Files/prism-4.6/bin/prism.bat"
load_ae = True

env = gym.make(env_name)

trace_file = f"{env_name}_{num_traces}_traces"
traces = load(trace_file)
if traces is None:
    dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
    traces = [get_traces_from_policy(dqn_agent, env, num_traces, action_map, randomness_probs=[0, 0.01, 0.025, 0.05])]
    save(traces, trace_file)
print('Traces obtained')

traces = traces[0]

# Model Initialization
autoencoder = AE()

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(autoencoder.parameters())

observations = list()
for trace in traces:
    for obs, _, _, _ in trace:
        observations.append(torch.tensor(obs))

train_loader = DataLoader(observations, batch_size=16)

if load_ae and os.path.exists('autoencoder.pickle'):
    print('Autoencoded loaded')
    with open('autoencoder.pickle', 'rb') as handle:
        autoencoder = pickle.load(handle)
else:
    train(autoencoder, train_loader, 100, optimizer, loss_function)
    with open('autoencoder.pickle', 'wb') as handle:
        pickle.dump(autoencoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

test_observations = random.choices(observations, k=3)
display_examples(autoencoder, test_observations)

# visualize_abstraction(autoencoder, traces)
# exit()

alergia_traces, goal_states = create_alergia_sequances(autoencoder, traces)

# action_dict = compute_history_based_prediction(alergia_traces, max_history_len=5)
#
# evaluate(env, action_dict, autoencoder, strategy='longest')

alergia_model = run_JAlergia(alergia_traces, 'mdp', path_to_jAlergia_jar='alergia.jar')

alergia_model.make_input_complete('sink_state')

remove_nan(alergia_model)
#alergia_model.save('test_ae_model')
#alergia_model.visualize()

test_with_scheduler(env, autoencoder, alergia_model, goal_states)

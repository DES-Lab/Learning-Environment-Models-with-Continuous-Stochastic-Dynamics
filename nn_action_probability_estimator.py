import os
from collections import defaultdict
from math import sqrt, log
from random import shuffle, random, choice
from statistics import mean

import numpy
from aalpy.automata import McState
from aalpy.base import SUL
from aalpy.learning_algs import run_JAlergia, run_Alergia, run_stochastic_Lstar
from aalpy.oracles import RandomWordEqOracle
from aalpy.utils import load_automaton_from_file
from scipy import stats

import gym
import torch
from sklearn.cluster import KMeans
from stable_baselines3 import DQN
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from agents import load_agent
from prism_scheduler import PrismInterface
from utils import save, load

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}


def get_observation_action_pairs(num_ep=10000):
    dqn_agent = load_agent("araffin/dqn-LunarLander-v2", 'dqn-LunarLander-v2.zip', DQN)

    sample_num = 0
    observation_actions_pairs = []
    for _ in range(num_ep):
        sample_num += 1
        if sample_num % 100 == 0:
            print(sample_num)
        obs = env.reset()
        while True:
            action, state = dqn_agent.predict(obs)
            observation_actions_pairs.append((obs, action))
            obs, reward, done, _ = env.step(action)
            if done:
                break

    return observation_actions_pairs


def evaluate_on_environment(env, model, num_episodes=100, render=False):
    all_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        ep_rew = 0
        while True:
            t = torch.tensor(obs, device=device)
            a = model(t)
            action = torch.argmax(a).data.item()
            obs, rew, done, info = env.step(action)
            if render:
                env.render()
            ep_rew += rew
            if done:
                all_rewards.append(ep_rew)
                break
    print('Mean reward:', mean(all_rewards))


def get_output(obs, model):
    t = torch.tensor(obs, requires_grad=False)
    t.unsqueeze_(dim=0)
    y = model(t)
    s = model.softmax(y)
    indices = [str(i) for i in numpy.argsort(s.tolist()[0], )]
    indices.reverse()
    return 'c' + ''.join(indices)


def get_samples_and_model_for_alergia(env, model, num_episodes=300, render=False):
    alergia_mc_samples = []
    for _ in range(num_episodes):
        if len(alergia_mc_samples) % 100 == 0:
            print(len(alergia_mc_samples))
        obs = env.reset()
        sample = ['INIT']
        while True:
            t = torch.tensor(obs, device=device)
            a = model(t)
            action = torch.argmax(a).data.item()
            obs, rew, done, info = env.step(action)
            if render:
                env.render()
            output = None
            if rew == 100:
                output = 'GOAL'
            output = output if output else get_output(obs, model)
            sample.append((action_map[int(action)], output))
            if done:
                break
        alergia_mc_samples.append(sample)

    # mc = run_JAlergia(alergia_mc_samples, 'mdp', 'alergia.jar')
    mc = run_Alergia(alergia_mc_samples, 'mdp')
    return mc


def test_mc(mc, env):
    all_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        mc.reset_to_initial()
        ep_rew = 0
        while True:
            action = int(mc.step())
            obs, rew, done, info = env.step(action)
            env.render()
            ep_rew += rew
            if done:
                all_rewards.append(ep_rew)
                break
    print('Mean reward:', mean(all_rewards))


class HoeffdingClustering():
    def __init__(self, nn, alpha=0.005):
        self.nn = nn
        self.alpha = alpha
        self.clusters = defaultdict(list)

    def are_different(self, c1, c2):
        # from scipy.stats import kstest
        # _, p = kstest(c1, c2)
        # return p <= 0.05
        # n1 = sum(c1)
        # n2 = sum(c2)
        #
        # if n1 > 0 and n2 > 0:
        import numpy
        return numpy.argsort(c1).tolist() == numpy.argsort(c2).tolist()
        # n1, n2 = 1, 1
        # for i in range(len(c1)):
        #     if abs(c1[i] / n1 - c2[i] / n2) > \
        #             ((sqrt(1 / n1) + sqrt(1 / n2)) * sqrt(0.5 * log(2 / self.alpha))):
        #         return True
        # return False

    def compute_clusters(self, observations):
        predictions = self.nn(observations)
        prediction_probabilities = self.nn.softmax(predictions)

        num_of_clusters = 1
        if not self.clusters:
            self.clusters[f'c{num_of_clusters}'].append(0)

        for i in range(100):
            current_observation = prediction_probabilities[i].data.tolist()
            cluster_found = False
            for cluster_label, cluster_element_indices in self.clusters.items():
                # OPTION TO CONSIDER WHOLE CLUSTER
                cluster_incompatible = False
                for index in cluster_element_indices:
                    cluster_element = prediction_probabilities[index].data.tolist()
                    if self.are_different(current_observation, cluster_element):
                        cluster_incompatible = True
                        break
                if not cluster_incompatible:
                    self.clusters[cluster_label].append(i)
                    cluster_found = True
                    break

            if not cluster_found:
                num_of_clusters += 1
                # print(num_of_clusters)
                self.clusters[f'c{num_of_clusters}'].append(i)

        print('Number of clusters', num_of_clusters)
        # for i in range(100):
        #     print(prediction_probabilities[i].data.tolist())


class ObservationActionsDataset(Dataset):
    def __init__(self, observations_action_pairs):
        self.observations_action_pairs = observations_action_pairs

    def __len__(self):
        return len(self.observations_action_pairs)

    def __getitem__(self, index):
        obs, action = self.observations_action_pairs[index]
        return torch.tensor(obs, device=device), torch.LongTensor([action.item()], device=device)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dimension, hidden_space_dimension, output_dimension):
        super(NeuralNetwork, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dimension, hidden_space_dimension),
            nn.ReLU(),
            nn.Linear(hidden_space_dimension, hidden_space_dimension // 2),
            nn.ReLU(),
            nn.Linear(hidden_space_dimension // 2, hidden_space_dimension // 2),
            nn.ReLU(),
            nn.Linear(hidden_space_dimension // 2, output_dimension),
            # nn.ReLU(),
            # nn.Linear(hidden_space_dimension // 4, output_dimension),
        ).to(device)

    def forward(self, x):
        return self.linear_relu_stack(x)
        # return self.softmax(self.get_prediction_probabilities(x))

    # def get_prediction_probabilities(self, x):
    # return self.softmax(x)


def train_nn(nn, optim, loss_criterion, train_data, test_data, num_epochs=100, stopping_accuracy=0.99):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        batch_loss = []
        nn.train()
        for i, data in enumerate(train_data):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optim.zero_grad()

            # forward + backward + optimize
            outputs = nn(inputs)
            labels = labels.squeeze_()  # TODO https://paperswithcode.com/task/imitation-learning
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optim.step()

            # print statistics
            batch_loss.append(loss.item())

        with torch.no_grad():
            nn.eval()
            total, correct = 0, 0
            for test_inputs, test_labels in test_data:
                output = nn(test_inputs)
                _, predictions = output.max(1)
                test_labels = test_labels.squeeze_()  # TODO CHECK THIS SQUEEZING

                correct += predictions.eq(test_labels).sum().item()
                total += predictions.size(0)

            accuracy = correct / total
            print('Epoch {0}: Accuracy: {1:2.2f}, Loss :{2:2.5f}'.format(epoch + 1, accuracy, mean(batch_loss)))
            if accuracy >= stopping_accuracy:
                break

    print('Finished Training')


def test_mdp(env, model, mdp):
    import aalpy.paths

    aalpy.paths.path_to_prism = 'C:/Program Files/prism-4.6/bin/prism.bat'

    prism_interface = PrismInterface('GOAL', mdp)
    for _ in range(100):
        obs = env.reset()
        prism_interface.reset()

        reward = 0

        t = torch.tensor(obs, device=device)
        a = model(t)
        initial_action = torch.argmax(a).data.item()
        obs, _, _, _ = env.step(initial_action)
        prism_interface.step_to(action_map[int(initial_action)], get_output(obs, model))

        while True:

            action = prism_interface.get_input()
            if action is None:
                t = torch.tensor(obs, device=device)
                a = model(t)
                action = int(torch.argmax(a).data.item())
                print('Cannot schedule an action')

            obs, rew, done, info = env.step(input_map[action])
            reward += rew

            label = get_output(obs, model)
            reached_state = prism_interface.step_to(action, label)
            env.render()

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


def active_learning(env, model):
    class ActiveSUL(SUL):
        def __init__(self, env, model):
            super().__init__()
            self.env = env
            self.model = model
            self.done = False
            self.obs = None

        def pre(self):
            self.done = False
            self.obs = self.env.reset()

        def post(self):
            pass

        def step(self, letter):
            t = torch.tensor(self.obs, device=device)
            a = model(t)
            action = torch.argmax(a).data.item()
            self.obs, rew, done, info = self.env.step(action)
            if self.done or done:
                self.done = True
                if rew >= 100:
                    return 'GOAL'
                return 'DONE'
            return action_map[int(action)]

    alph = ['step']
    sul = ActiveSUL(env, model)
    eq_oracle = RandomWordEqOracle(alph, sul)

    model = run_stochastic_Lstar(alph, sul, eq_oracle, automaton_type='smm', min_rounds=3, max_rounds=10)
    model.save('active_model')
    model.visualize()


env = gym.make("LunarLander-v2")

num_episodes = 1000
load_observations = True
load_nn_if_exists = True
hidden_size = 64  # second layer is hidden_size /2

if load_observations and os.path.exists(f'obs_actions_pairs_{num_episodes}.pickle'):
    obs_action_pairs = load(f'obs_actions_pairs_{num_episodes}.pickle')
    if obs_action_pairs:
        print('Observation actions pairs loaded')
else:
    print('Computing observation action pairs')
    obs_action_pairs = get_observation_action_pairs(num_episodes)
    save(obs_action_pairs, path=f'obs_actions_pairs_{num_episodes}')

shuffle(obs_action_pairs)

training_ratio, len_dataset = 0.8, len(obs_action_pairs)
split_index = int(len_dataset * training_ratio)
training_data = ObservationActionsDataset(obs_action_pairs[:split_index])
test_data = ObservationActionsDataset(obs_action_pairs[split_index:])

train_data_loader = DataLoader(training_data, batch_size=128, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=128, shuffle=True)

neural_network = NeuralNetwork(input_dimension=8, hidden_space_dimension=hidden_size, output_dimension=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(neural_network.parameters(), lr=0.01, )

model_path = f'nn_{hidden_size}.pt'
if not load_nn_if_exists or not os.path.exists(model_path):
    train_nn(neural_network, optimizer, criterion, train_data_loader, test_data_loader, num_epochs=10,
             stopping_accuracy=0.98)
    torch.save(neural_network.state_dict(), model_path)
    print(f'Weights saved to {model_path}.')
else:
    neural_network.load_state_dict(torch.load(model_path))
    print('NN weights loaded.')

neural_network.eval()

# mdp = get_samples_and_model_for_alergia(env, neural_network, num_episodes=500, render=False)
# mdp.save()

# model = load_automaton_from_file('active_model.dot', 'smm')
# model.visualize()
active_learning(env, neural_network)

#
# mdp = load_automaton_from_file('LearnedModel.dot', 'mdp')
# # mdp.visualize()
# mdp.make_input_complete()
# test_mdp(env, neural_network, mdp)
# # test_mc(mdp, env)

exit()
evaluate_on_environment(env, neural_network, render=False)

hoeffding_clustering = HoeffdingClustering(neural_network)
observations = torch.stack([d[0] for d in train_data_loader.dataset])

hoeffding_clustering.compute_clusters(observations)

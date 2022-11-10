import os
from random import shuffle
from statistics import mean

import gym
import torch
from stable_baselines3 import DQN
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from agents import load_agent
from utils import save, load

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


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


class ObservationActionsDataset(Dataset):
    def __init__(self, observations_action_pairs):
        self.observations_action_pairs = observations_action_pairs

    def __len__(self):
        return len(self.observations_action_pairs)

    def __getitem__(self, index):
        obs, action = self.observations_action_pairs[index]
        return torch.tensor(obs, device=device), torch.tensor(action, device=device)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dimension, hidden_space_dimension, output_dimension):
        super(NeuralNetwork, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dimension, hidden_space_dimension),
            nn.ReLU(),
            nn.Linear(hidden_space_dimension, output_dimension),
        ).to(device)

    def forward(self, x):
        return self.linear_relu_stack(x)

    def get_prediction_probabilities(self, x):
        return self.softmax(x)


def train_nn(nn, optim, loss_criterion, train_data, test_data, num_epochs=100, stopping_accuracy=0.95):
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
                _, predictions = output.data.cpu().topk(1, dim=1)
                correct += predictions.eq(test_labels).sum().item()
                total += predictions.size(0)

            accuracy = correct / total
            print('Epoch {0}: Accuracy: {1:2.2f}, Loss :{2:2.5f}'.format(epoch, accuracy, mean(batch_loss)))

    print('Finished Training')


env = gym.make("LunarLander-v2")

num_episodes = 1000
load_observations = True

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

train_data_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True)

neural_network = NeuralNetwork(input_dimension=8, hidden_space_dimension=64, output_dimension=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(neural_network.parameters(), lr=0.001)

train_nn(neural_network, optimizer, criterion, train_data_loader, test_data_loader)

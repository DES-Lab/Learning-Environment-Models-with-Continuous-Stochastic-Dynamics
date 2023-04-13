import os
import pickle
from statistics import mean

import aalpy.paths
import gym
import torch
from stable_baselines3 import DQN
from torch.utils.data import DataLoader

from agents import load_agent
from utils import load, get_traces_from_policy, save

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}


class AE(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, latent_dim),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_autoencoder(self, data, epochs, batch_size):

        observations = [torch.tensor(obs) for obs in data]

        train_loader = DataLoader(observations, batch_size=batch_size)

        # Validation using MSE Loss function
        loss_function = torch.nn.MSELoss()

        # Using an Adam Optimizer with lr = 0.1
        optimizer = torch.optim.Adam(self.parameters())

        print(f'Training autoencoder for {epochs} epochs.')

        for epoch in range(epochs):

            losses = []

            for obs in train_loader:
                reconstructed = self(obs)

                # Calculating the loss function
                loss = loss_function(obs, reconstructed)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Storing the losses in a list for plotting
                losses.append(loss.item())

            print(f'Epoch {epoch}, loss : {mean(losses)}')

    def display_examples(self, data):
        for obs in data:
            reconstructed = self(obs)
            print('---------------------')
            print(obs.detach().numpy()[0].tolist())
            print(reconstructed.detach().numpy()[0].tolist())
            print('Latent space', self.encoder(obs))


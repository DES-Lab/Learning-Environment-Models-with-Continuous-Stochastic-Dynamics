import random

from statistics import mean

import gym
import torch
import aalpy.paths
from aalpy.learning_algs import run_Alergia, run_JAlergia
from stable_baselines3 import DQN
from torch.utils.data import DataLoader
from tqdm import tqdm

from agents import load_agent
from prism_scheduler import PrismInterface
from utils import load, get_traces_from_policy, save

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}

env_name = "LunarLander-v2"

num_traces = 100
aalpy.paths.path_to_prism = "C:/Program Files/prism-4.6/bin/prism.bat"


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
        out_str = str(rounded).replace('.','_')
        if out < 0:
            out_str = str(abs(rounded)).replace('.', '_')
            return f'sn{out_str}'
        return f's{out_str}'


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
    alergia_sequances = []
    print('Creating Alergia sequances')
    for trace in tqdm(traces):
        seq = ['INIT']
        for obs, action, reward, done in trace:
            abstract_obs = model.encode(torch.tensor(obs)) if not done else 'DONE'
            seq.extend((action_map[int(action)], abstract_obs))

        alergia_sequances.append(seq)
    return alergia_sequances


def test_with_scheduler(model, alergia_model, env, num_episodes=100):
    prism_interface = PrismInterface('DONE', alergia_model)
    scheduler = prism_interface.scheduler

    total_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        scheduler.reset()
        # hack to reach first state in the model
        initial_action = 'no_action'
        obs, rew, done, _ = env.step(input_map[initial_action])
        abstract_obs = model.encode(torch.tensor(obs))
        scheduler.step_to(initial_action, abstract_obs)

        ep_rew = 0

        while True:
            action = scheduler.get_input()
            obs, rew, done, _ = env.step(input_map[action])
            env.render()

            abstract_obs = model.encode(torch.tensor(obs))
            scheduler.step_to(action, abstract_obs)
            ep_rew += rew

            if done:
                print("Episode reward", ep_rew)
                total_rewards.append(ep_rew)
                break


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

train_loader = DataLoader(observations, batch_size=32)

train(autoencoder, train_loader, 50, optimizer, loss_function)

test_observations = random.choices(observations, k=3)

display_examples(autoencoder, test_observations)

alergia_traces = create_alergia_sequances(autoencoder, traces)
alergia_model = run_JAlergia(alergia_traces, 'mdp', path_to_jAlergia_jar='alergia.jar')
alergia_model.make_input_complete('self_loop')
alergia_model.visualize()

test_with_scheduler(autoencoder, alergia_model, env)

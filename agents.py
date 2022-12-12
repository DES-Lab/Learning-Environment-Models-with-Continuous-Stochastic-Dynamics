import sys

import gym
from huggingface_sb3 import load_from_hub
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecExtractDictObs


def load_agent(repo_id, file_name, alg):
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    checkpoint = load_from_hub(
        repo_id=repo_id,
        filename=file_name,
    )

    model = alg.load(checkpoint, custom_objects=custom_objects)
    return model


def evaluate_agent(model, model_name, env):
    mean_reward, std_reward = evaluate_policy(model, env, render=False, n_eval_episodes=100,
                                              deterministic=True, warn=False)
    print(f"{model_name} mean_reward={mean_reward:.2f} +/- {std_reward}")


def get_lunar_lander_agents(evaluate=False):
    dqn_agent = load_agent('araffin/ppo-LunarLander-v2', 'ppo-LunarLander-v2.zip', A2C)
    a2c_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
    agents = [('dqn', dqn_agent), ('a2c', a2c_agent)]

    if evaluate:
        print('Evaluating agents')
        lunar_lander_env = gym.make('LunarLander-v2')
        for name, agent in agents:
            evaluate_agent(agent, name, lunar_lander_env)

    return agents


class RacingCarWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = gym.wrappers.resize_observation.ResizeObservation(shape=64)
        next_state = gym.wrappers.gray_scale_observation.GrayScaleObservation(keep_dim=True)

        next_state

        return next_state, reward, done, info


def get_car_racing_agents(evaluate=False):
    ppo_recurrent = load_agent('NikitaBaramiia/PPO-CarRacing-v0', 'ppo-CarRacing-v0.zip', PPO)
    ppo_recurrent2 = load_agent('sb3/ppo_lstm-CarRacing-v0', 'ppo_lstm-CarRacing-v0.zip', PPO)

    agents = [('ppo_recurrent1', ppo_recurrent), ('ppo_recurrent2', ppo_recurrent2), ]

    if evaluate:
        if evaluate:
            print('Evaluating agents')

            env = gym.make('CarRacing-v0')
            env = gym.wrappers.ResizeObservation(env, 64)
            env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)

            for name, agent in agents:
                # TODO does not work
                # ValueError: Error: Unexpected observation shape (1, 96, 96, 3) for Box environment, please use (2, 64, 64) or (n_env, 2, 64, 64) for the observation shape.
                evaluate_agent(agent, name, env)
        # 21
        return agents


def get_lunar_lander_agents_smc(evaluate=False):
    a2c_agent = load_agent('araffin/ppo-LunarLander-v2', 'ppo-LunarLander-v2.zip', A2C)
    dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
    agents = [('dqn', dqn_agent), ('a2c', a2c_agent)]

    lunar_lander_env = gym.make('LunarLander-v2')

    if evaluate:
        print('Evaluating agents')
        for name, agent in agents:
            evaluate_agent(agent, name, lunar_lander_env)

    available_actions = [0, 1, 2, 3]
    return agents, available_actions, lunar_lander_env


def get_cartpole_agents_smc(evaluate=False):
    ppo_agent = load_agent('sb3/ppo-CartPole-v1', 'ppo-CartPole-v1.zip', PPO)
    dqn_agent = load_agent('sb3/dqn-CartPole-v1', 'dqn-CartPole-v1.zip', DQN)
    agents = [('ppo', ppo_agent), ('dqn', dqn_agent)]

    env = gym.make('CartPole-v1')

    if evaluate:
        print('Evaluating agents')
        for name, agent in agents:
            evaluate_agent(agent, name, env)

    available_actions = [0, 1]
    return agents, available_actions, env


def get_mountaincar_agents_smc(evaluate=False):
    dqn_agent = load_agent('sb3/dqn-MountainCar-v0', 'dqn-MountainCar-v0.zip', DQN)
    dqn_agent2 = load_agent('DBusAI/DQN-MountainCar-v0', 'DQN-MountainCar-v0.zip', DQN)
    agents = [('dqn1', dqn_agent), ('dqn2', dqn_agent2)]

    env = gym.make('MountainCar-v0')

    if evaluate:
        print('Evaluating agents')
        for name, agent in agents:
            evaluate_agent(agent, name, env)

    available_actions = [0, 1, 2]
    return agents, available_actions, env


if __name__ == '__main__':
    get_car_racing_agents(evaluate=True)

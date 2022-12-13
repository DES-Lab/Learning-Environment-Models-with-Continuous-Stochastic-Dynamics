import gym
from stable_baselines3 import A2C, DQN, PPO, SAC, DDPG
from sb3_contrib import TQC
from stable_baselines3.common.type_aliases import Schedule

from agents import load_agent, evaluate_agent


def get_car_racing_agents(evaluate=False):
    # ppo_recurrent = load_agent('NikitaBaramiia/PPO-CarRacing-v0', 'ppo-CarRacing-v0.zip', PPO)
    # ppo_recurrent2 = load_agent('sb3/ppo_lstm-CarRacing-v0', 'ppo_lstm-CarRacing-v0.zip', PPO)
    ppo = load_agent('meln1k/ppo-CarRacing-v0', 'ppo-CarRacing-v0.zip', PPO)

    agents = [('ppo', ppo), ]  # ('ppo_recurrent2', ppo_recurrent2), ]

    if evaluate:
        if evaluate:
            print('Evaluating agents')

            env = gym.make('CarRacing-v0')
            # env = gym.wrappers.ResizeObservation(env, 64)
            # env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)

            for name, agent in agents:
                evaluate_agent(agent, name, env)

        return agents


def get_bipedal_walker_agents(evaluate=False):
    sac_agent = load_agent('format37/BipedalWalker-v3', 'SAC-Mlp.zip', SAC)
    ddpg_agent = load_agent('sb3/ddpg-BipedalWalker-v3', 'ddpg-BipedalWalker-v3.zip', DDPG)
    a2c_agent = load_agent('sb3/a2c-BipedalWalker-v3', 'a2c-BipedalWalker-v3.zip', A2C)
    tqc_agent = load_agent('sb3/tqc-BipedalWalker-v3', 'tqc-BipedalWalker-v3.zip', TQC)

    env = gym.make("BipedalWalker-v3")

    agents = [('sac', sac_agent), ('ddpg_agent', ddpg_agent), ('a2c_agent', a2c_agent), ('tqc_agent', tqc_agent)]

    if evaluate:
        print('Evaluating agents')
        for name, agent in agents:
            evaluate_agent(agent, name, env)

    return agents, 'bipedal_walker', env


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
    ppo_agent = load_agent('vukpetar/ppo-MountainCar-v0', 'ppo-mountaincar-v0.zip', PPO)
    ppo_agent2 = load_agent('format37/PPO-MountainCar-v0', 'PPO-Mlp.zip', PPO)

    agents = [('dqn1', dqn_agent), ('dqn2', dqn_agent2), ('ppo1', ppo_agent), ('ppo2', ppo_agent2)]

    env = gym.make('MountainCar-v0')

    if evaluate:
        print('Evaluating agents')
        for name, agent in agents:
            evaluate_agent(agent, name, env)

    available_actions = [0, 1, 2]
    return agents, available_actions, env



if __name__ == '__main__':
    get_lunar_lander_agents_smc(evaluate=True)

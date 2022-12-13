import gym
from sb3_contrib import TQC, ARS
from stable_baselines3 import A2C, DQN, PPO, SAC, DDPG

from agents import load_agent, evaluate_agent


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

    def evaluate_obs(obs, rew, done, info, results_dict):
        if rew == -100:
            results_dict['crash'] += 1
        elif rew == 300:
            results_dict['goal'] += 1

    return agents, 'continuous', env, evaluate_obs


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

    def evaluate_obs(obs, rew, done, info, results_dict):
        if rew == 100:
            results_dict['goal'] += 1
        elif rew == -100:
            results_dict['crash'] += 1
        else:
            results_dict['timeout'] += 1

    return agents, available_actions, lunar_lander_env, evaluate_obs


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

    def evaluate_obs(obs, rew, done, info, results_dict):
        pass

    return agents, available_actions, env, evaluate_obs


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

    def evaluate_obs(obs, rew, done, info, results_dict):
        if info['ep_len'] == 200:
            results_dict['timeout'] += 1

    return agents, available_actions, env, evaluate_obs


def get_walker2d_agents_smc(evaluate=False):
    tqc_agent = load_agent('sb3/tqc-Walker2d-v3', 'tqc-Walker2d-v3.zip', TQC)
    ars_agent = load_agent('sb3/ars-Walker2d-v3', 'ars-Walker2d-v3.zip', ARS)

    agents = [('tqc_agent', tqc_agent), ('ars_agent', ars_agent), ()]
    env = gym.make('Walker2d-v3')

    if evaluate:
        print('Evaluating agents')
        for name, agent in agents:
            evaluate_agent(agent, name, env)

    available_actions = 'continuous'

    def evaluate_obs(obs, rew, done, info, results_dict):
        if info['ep_len'] == 1000:
            results_dict['goal'] += 1
        else:
            results_dict['crash'] += 1

    return agents, available_actions, env, evaluate_obs



if __name__ == '__main__':
    get_walker2d_agents_smc(evaluate=True)



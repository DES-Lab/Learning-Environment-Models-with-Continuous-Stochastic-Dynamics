import gym
from huggingface_sb3 import load_from_hub
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy


def load_agent(repo_id, file_name, alg):
    checkpoint = load_from_hub(
        repo_id=repo_id,
        filename=file_name,
    )
    model = alg.load(checkpoint)
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


if __name__ == '__main__':
    model1 = load_agent('araffin/ppo-LunarLander-v2', 'ppo-LunarLander-v2.zip', A2C)
    model2 = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)

    env = gym.make('LunarLander-v2', )
    evaluate_agent(model1, env)
    evaluate_agent(model2, env)

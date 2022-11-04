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


def evaluate_agent(model, env):
    mean_reward, std_reward = evaluate_policy(model, env, render=False, n_eval_episodes=100,
                                              deterministic=True, warn=False)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


if __name__ == '__main__':
    model1 = load_agent('araffin/ppo-LunarLander-v2', 'ppo-LunarLander-v2.zip', A2C)
    model2 = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)

    env = gym.make('LunarLander-v2', )
    evaluate_agent(model1, env)
    evaluate_agent(model2, env)

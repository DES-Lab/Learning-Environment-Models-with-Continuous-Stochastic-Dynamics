import sys

import gym
from huggingface_sb3 import load_from_hub
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy


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


def evaluate_agent(model, model_name, env, render=False):
    mean_reward, std_reward = evaluate_policy(model, env, render=render, n_eval_episodes=100,
                                              deterministic=True, warn=False)
    print(f"{model_name} mean_reward={mean_reward:.2f} +/- {std_reward}")


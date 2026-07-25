from huggingface_sb3 import load_from_hub
from stable_baselines3.common.evaluation import evaluate_policy


def load_agent(repo_id, file_name, alg, env=None):
    """Load a pretrained SB3 agent from the HuggingFace hub.

    The pretrained checkpoints on the hub were saved with stable-baselines3 1.x,
    which pickled the observation/action spaces as `gym` objects. Since we run on
    gymnasium, those pickles are unreadable, so the spaces are overridden with the
    ones of `env`. Pass the environment the agent will act in; if it is omitted,
    loading such legacy checkpoints will fail with `No module named 'gym'`.
    """
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    if env is not None:
        custom_objects["observation_space"] = env.observation_space
        custom_objects["action_space"] = env.action_space

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

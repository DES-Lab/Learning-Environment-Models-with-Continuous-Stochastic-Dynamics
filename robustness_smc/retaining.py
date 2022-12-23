from typing import Optional, Tuple

import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.utils import is_vectorized_observation

from agents import load_agent, evaluate_agent
from collections import OrderedDict


class RobustDQN(DQN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.policy_steps = 5
        self.random_steps = 1

        self.policy_counter, self.random_counter = 0, 0

    # def predict(self, observation: np.ndarray,
    #             state: Optional[Tuple[np.ndarray, ...]] = None,
    #             episode_start: Optional[np.ndarray] = None,
    #             deterministic: bool = False, ):
    #
    #     if not deterministic and self.policy_counter >= self.policy_steps:
    #         self.random_counter += 1
    #         if self.random_counter == self.random_steps:
    #             self.policy_counter = 0
    #             self.random_counter = 0
    #
    #         if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
    #             if isinstance(self.observation_space, gym.spaces.Dict):
    #                 n_batch = observation[list(observation.keys())[0]].shape[0]
    #             else:
    #                 n_batch = observation.shape[0]
    #             action = np.array([self.action_space.sample() for _ in range(n_batch)])
    #         else:
    #             action = np.array(self.action_space.sample())
    #     else:
    #         self.policy_counter += 1
    #         action, state = self.policy.predict(observation, state, episode_start, deterministic)
    #
    #     return action, state


def retrain_lunar_lander_agent():
    lunar_lander_env = gym.make('LunarLander-v2')

    lunar_lander_env = gym.make('LunarLander-v2')
    dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)

    robust_agent = RobustDQN(policy='MlpPolicy', policy_kwargs=dict(net_arch=[256, 256]), env=lunar_lander_env, exploration_initial_eps=0.7, exploration_final_eps=0.4)
    robust_agent.set_parameters(dqn_agent.get_parameters())

    evaluate_agent(robust_agent, 'Original', lunar_lander_env)
    robust_agent.learn(10000, progress_bar=True)

    evaluate_agent(robust_agent, 'Retrained', lunar_lander_env)

    from robustness_smc.smc import smc
    smc(robust_agent, lunar_lander_env, 10, 3, 'robust', [0, 1, 2, 3])


if __name__ == '__main__':
    retrain_lunar_lander_agent()

import gym
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, RGBImgObsWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN
import gym


class MiniWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self,action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info

env = gym.make("MiniGrid-Empty-5x5-v0") #, render_mode="human")
minigrid_env = env
env = RGBImgObsWrapper(env) # Get pixel observations
env = ImgObsWrapper(env) # Get rid of the 'mission' field
env = MiniWrapper(env)

# print(env.reset())
# print(env.step(1))

#check_env(env)

model = DQN("CnnPolicy", env, verbose=1,device="cpu")
model.learn(total_timesteps=100_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    print("FOO")
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    print(reward, ":", done)
    # vec_env.render(mode="human")
    # VecEnv resets automatically
    if done:
      obs = vec_env.reset()

env.close()

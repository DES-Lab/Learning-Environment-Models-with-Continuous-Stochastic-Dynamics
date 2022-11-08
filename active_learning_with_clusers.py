import gym
from aalpy.base import SUL
from aalpy.learning_algs import run_stochastic_Lstar
from aalpy.oracles import RandomWordEqOracle

from utils import load


class ClusterSUL(SUL):
    def __init__(self, env, cf):
        super().__init__()
        self.env = env
        self.cf = cf
        self.done = False
        self.x = 0

    def pre(self):
        self.done = False
        self.env.reset()

    def post(self):
        pass

    def step(self, letter):
        for _ in range(10):
            obs, rew, done, info = self.env.step(letter)
        self.x += 1
        if self.x >= 20000:
            self.env.render()
        if done and rew == 100:
            self.done=True
            return 'GOAL'
        if done or self.done:
            self.done = True
            return 'DONE'
        return f'c{self.cf.predict(obs.reshape(1, -1))}'


clustering_function = load('k_means_16.pickle')
env = gym.make('LunarLander-v2', )
sul = ClusterSUL(env, clustering_function)
input_alphabet = [0, 1, 2, 3]

eq_oracle = RandomWordEqOracle(input_alphabet, sul, num_walks=1000, min_walk_len=3, max_walk_len=50)
learned_model = run_stochastic_Lstar(input_alphabet, sul, eq_oracle, automaton_type='smm')

learned_model.save('active_stochastic')

import matplotlib.pyplot as plt
import numpy as np

from utils import load


def visualize_results(data):
    refinement_rounds = list(data.keys())
    all_rewards = [i['reward'] for k, i in data.items()]
    mean_rew = np.array([i[0] for i in all_rewards])
    std_dev_rew = np.array([i[1] for i in all_rewards])

    plt.plot(refinement_rounds, mean_rew, 'r-', label='Mean Reward')
    plt.fill_between(refinement_rounds, mean_rew - std_dev_rew, mean_rew + std_dev_rew, color='b', alpha=0.2)

    plt.xlabel('Refinement Round')
    plt.ylabel('Reward')
    plt.title(f'Mountain Car: {len(refinement_rounds)} Iterations')

    plt.show()


if __name__ == '__main__':
    data = load('pickles/results/MountainCar-v0_MountainCar-v0_num_traces_2500_powerTransformer_n_clusters_128.pk')
    visualize_results(data)

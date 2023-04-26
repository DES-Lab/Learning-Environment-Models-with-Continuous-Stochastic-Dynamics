import matplotlib.pyplot as plt
import numpy as np

from utils import load


def visualize_rewards(data, env_name):
    refinement_rounds = list(data.keys())
    all_rewards = [i['reward'] for k, i in data.items()]
    mean_rew = np.array([i[0] for i in all_rewards])
    std_dev_rew = np.array([i[1] for i in all_rewards])

    plt.plot(refinement_rounds, mean_rew, 'r-', label='Mean Reward')
    plt.fill_between(refinement_rounds, mean_rew - std_dev_rew, mean_rew + std_dev_rew, color='b', alpha=0.2)

    plt.xlabel('Refinement Round')
    plt.ylabel('Reward')
    plt.legend()
    plt.title(f'{env_name}: {len(refinement_rounds)} Iterations of 25 Episodes')

    plt.show()


def visualize_goal_and_crash(data, env_name):
    refinement_rounds = list(data.keys())
    goal_reached = [i['goal_reached'] for k, i in data.items()]
    goal_reached_percentage = [i['goal_reached_percentage'] for k, i in data.items()]
    crash_reached = [i['crash'] for k, i in data.items()]

    plt.plot(refinement_rounds, crash_reached, 'r-', label='Crash')
    plt.plot(refinement_rounds, goal_reached, 'g-',  label='Goal')
    # plt.plot(refinement_rounds, goal_reached_percentage, 'r-', label='Goal %')

    plt.xlabel('Refinement Round')
    plt.ylabel('Frequency in Refinement Round')
    plt.legend()
    plt.title(f'{env_name}: {len(refinement_rounds)} Iterations of 25 Episodes')

    plt.show()


if __name__ == '__main__':
    # data = load('pickles/results/MountainCar-v0_MountainCar-v0_num_traces_2500_powerTransformer_n_clusters_128.pk')
    # data = load('pickles/results/Acrobot-v1_num_traces_2500_scaler_lda_2_n_clusters_128.pk')
    data = load('pickles/results/LunarLander-v2_num_traces_1000_manualMapper_powerTransformer_n_clusters_128_ri_100_ep_50.pk')
    visualize_rewards(data, 'LunarLander')
    visualize_goal_and_crash(data, 'LunarLander')

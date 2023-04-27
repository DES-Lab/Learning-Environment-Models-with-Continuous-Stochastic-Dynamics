from collections import defaultdict
from statistics import stdev, mean

import matplotlib.pyplot as plt
import numpy as np

from utils import load


def visualize_rewards_multiple_exp(experiments, env_name, baseline_val=None):
    refinement_rounds = []
    rewards_per_round = defaultdict(list)

    for data in experiments:
        refinement_rounds.append(max(list(data.keys())))
        for k, i in data.items():
            rewards_per_round[k].append(i['reward'][0])  # 0 is mean reward
        # std_dev_rew = np.array([i[1] for i in all_rewards])

    if len(set(refinement_rounds)) != 1:
        print(f'Experiments have different number of refinement rounds: {refinement_rounds}')
        print(f'Cutting all experiments to {min(refinement_rounds)}')
        min_rr, max_rr = min(refinement_rounds), max(refinement_rounds)
        for to_remove in range(min_rr + 1, max_rr + 1):
            if to_remove in rewards_per_round.keys():
                rewards_per_round.pop(to_remove)
        refinement_rounds = min_rr

    mean_rew = np.array([mean(i) for i in rewards_per_round.values()])
    std_dev_rew = np.array([stdev(i) for i in rewards_per_round.values()])

    refinement_rounds = list(range(1, refinement_rounds + 2))

    plt.plot(refinement_rounds, mean_rew, 'r-', label='Mean Reward')
    plt.fill_between(refinement_rounds, mean_rew - std_dev_rew, mean_rew + std_dev_rew, color='b', alpha=0.2)

    if baseline_val is not None:
        plt.plot(refinement_rounds, [baseline_val] * len(refinement_rounds), 'g-', label='RL baseline')

    plt.xlabel('Refinement Round')
    plt.ylabel('Reward')
    plt.legend()

    episodes_per_iter = experiments[0][0]["episodes_per_iteration"]
    plt.title(f'{env_name}: {len(refinement_rounds)} Iterations of {episodes_per_iter} Episodes')

    plt.show()


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
    plt.plot(refinement_rounds, goal_reached, 'g-', label='Goal')
    # plt.plot(refinement_rounds, goal_reached_percentage, 'r-', label='Goal %')

    plt.xlabel('Refinement Round')
    plt.ylabel('Frequency in Refinement Round')
    plt.legend()
    plt.title(f'{env_name}: {len(refinement_rounds)} Iterations of 25 Episodes')

    plt.show()


if __name__ == '__main__':
    # Cartpole experiments
    cartpole_files = ['pickles/results/exp_CartPole-v1_num_traces_2500_powerTransformer_n_clusters_128_ri_50_ep_50.pk',
                      'pickles/results/exp2_CartPole-v1_num_traces_1000_powerTransformer_n_clusters_128_ri_12_ep_50.pk',
                      'pickles/results/exp3_CartPole-v1_num_traces_2500_powerTransformer_n_clusters_128_ri_12_ep_50.pk']

    acrobot_files = [
        'pickles/results/exp0_Acrobot-v1_num_traces_2500_powerTransformer_lda_2_n_clusters_128_ri_25_ep_50.pk'
        , 'pickles/results/exp1_Acrobot-v1_num_traces_2500_powerTransformer_lda_2_n_clusters_128_ri_25_ep_50.pk'
        , 'pickles/results/exp2_Acrobot-v1_num_traces_2500_powerTransformer_lda_2_n_clusters_128_ri_25_ep_50.pk'
        , 'pickles/results/exp3_Acrobot-v1_num_traces_2500_powerTransformer_lda_2_n_clusters_128_ri_25_ep_50.pk'
        , 'pickles/results/exp4_Acrobot-v1_num_traces_2500_powerTransformer_lda_2_n_clusters_128_ri_25_ep_50.pk']

    mountain_car_files = [
        'pickles/results/exp0_MountainCar-v0_num_traces_2500_powerTransformer_n_clusters_128_ri_25_ep_50.pk'
        , 'pickles/results/exp1_MountainCar-v0_num_traces_2500_powerTransformer_n_clusters_128_ri_25_ep_50.pk'
        , 'pickles/results/exp2_MountainCar-v0_num_traces_2500_powerTransformer_n_clusters_128_ri_25_ep_50.pk']

    # data = load('pickles/results/MountainCar-v0_MountainCar-v0_num_traces_2500_powerTransformer_n_clusters_128.pk')
    # data = load('pickles/results/Acrobot-v1_num_traces_2500_scaler_lda_2_n_clusters_128.pk')
    # data = load('pickles/results/LunarLander-v2_num_traces_1000_manualMapper_powerTransformer_n_clusters_128_ri_100_ep_50.pk')
    # for f in acrobot_files:
    #     data = load(f)
    #     visualize_rewards(data, 'LunarLander')
    # visualize_goal_and_crash(data, 'LunarLander')

    visualize_rewards(load('pickles/results/exp0_MountainCar-v0_num_traces_2500_powerTransformer_n_clusters_256_ri_25_ep_50.pk'), 'MC')
    # baseline_values = {'MountainCar': -105, 'Acrobot': - 100, 'Cartpole': 200}
    #
    # all_acrobat_data = [load(d) for d in cartpole_files]
    # visualize_rewards_multiple_exp(all_acrobat_data, 'Cartpole', baseline_val=None)

import os

from collections import defaultdict
from statistics import stdev, mean, median, quantiles

import matplotlib.pyplot as plt
import numpy as np

from utils import load


def tikzplotlib_fix_ncols(obj):
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


#     iteration_round_data = defaultdict(list)
#     goal_reached_per = defaultdict(list)
#     tot_sum = []
#     for e in experiments:
#         ts = 0
#         for iter_round, data in e.items():
#             iteration_round_data[iter_round].extend(data['all_rewards'])
#             gp = sum([1 if i >= 195 else 0 for i in data['all_rewards']])
#             ts += sum(data['all_rewards'])
#             goal_reached_per[iter_round].append(gp)
#         tot_sum.append(ts)
#
#     print(tot_sum)
#     print(mean(tot_sum), stdev(tot_sum))
#
#     exit()

def get_iteration_averages(experiments, method):
    assert method in {'mean_stddev', 'median_quantiles'}
    if method == 'median_quantiles':
        f1, f2 = median, quantiles
    else:
        f1, f2 = mean, stdev

    iteration_round_data = defaultdict(list)
    for e in experiments:
        for iter_round, data in e.items():
            iteration_round_data[iter_round].extend(data['all_rewards'])

    for i in iteration_round_data.keys():
        removed_dead_state_big_penalty = []
        for r in iteration_round_data[i]:
            if r < -3000:
                removed_dead_state_big_penalty.append(-500)
            else:
                removed_dead_state_big_penalty.append(r)
        iteration_round_data[i] = removed_dead_state_big_penalty

    iteration_means, iteration_quantiles = dict(), dict()
    for i, values in iteration_round_data.items():
        iteration_means[i] = f1(values)
        iteration_quantiles[i] = f2(values)
        if method == 'median_quantiles':
            iteration_quantiles[i] = iteration_quantiles[i][0] - iteration_quantiles[i][2]

    return iteration_means, iteration_quantiles


def visualize_experiment_runs(experiments, env_name, method, baseline_val=None):
    plot_value_1, plot_value_2 = get_iteration_averages(experiments, method)
    plot_value_1 = np.array(list(plot_value_1.values()))
    plot_value_2 = np.array(list(plot_value_2.values()))

    refinement_rounds = list(range(1, len(experiments[0].keys()) + 1))

    fig = plt.figure()

    plt.plot(refinement_rounds, plot_value_1, 'r-', label='Mean Reward')
    plt.fill_between(refinement_rounds, plot_value_1 - plot_value_2, plot_value_1 + plot_value_2, color='r', alpha=0.2)

    if baseline_val is not None:
        plt.plot(refinement_rounds, [baseline_val] * len(refinement_rounds), 'g-', label='RL baseline')

    plt.xlabel('Refinement Round')
    plt.ylabel('Reward')
    plt.legend()

    episodes_per_iter = experiments[0][0]["episodes_per_iteration"]
    plt.title(f'{env_name}: {len(refinement_rounds)} Iterations of {episodes_per_iter} Episodes')

    plt.show()
    # import tikzplotlib
    # tikzplotlib_fix_ncols(fig)
    # tikzplotlib.save("figures/tmp.tex")


def visualize_multiple_experiments(experiment_1, experiment_2, env_name, method, baseline_val=None):
    exp_1_name, exp_2_name = experiment_1[0], experiment_2[0]

    exp_1_plot_val_1, exp_1_plot_val_2 = get_iteration_averages(experiment_1[1], method)
    exp_1_plot_val_1 = np.array(list(exp_1_plot_val_1.values()))
    exp_1_plot_val_2 = np.array(list(exp_1_plot_val_2.values()))

    exp_2_plot_val_1, exp_2_plot_val_2 = get_iteration_averages(experiment_2[1], method)
    exp_2_plot_val_1 = np.array(list(exp_2_plot_val_1.values()))
    exp_2_plot_val_2 = np.array(list(exp_2_plot_val_2.values()))

    refinement_rounds = list(range(1, len(experiment_1[1][0].keys()) + 1))

    fig = plt.figure()

    plt.plot(refinement_rounds, exp_1_plot_val_1, 'r-', label=f'Mean Reward: {exp_1_name}')
    plt.fill_between(refinement_rounds, exp_1_plot_val_1 - exp_1_plot_val_2, exp_1_plot_val_1 + exp_1_plot_val_2,
                     color='r', alpha=0.2)

    plt.plot(refinement_rounds, exp_2_plot_val_1, 'b-', label=f'Mean Reward: {exp_2_name}')
    plt.fill_between(refinement_rounds, exp_2_plot_val_1 - exp_2_plot_val_2, exp_2_plot_val_1 + exp_2_plot_val_2,
                     color='b', alpha=0.2)

    if env_name == 'Lunar Lander':
        # plt.plot(refinement_rounds, [baseline_val] * len(refinement_rounds), 'g-', label='RL baseline')
        plt.plot(refinement_rounds, [136] * len(refinement_rounds), 'gd', label='sb3-dqn')
        plt.plot(refinement_rounds, [181] * len(refinement_rounds), 'gx', label='sb3-a2c')
        plt.plot(refinement_rounds, [223] * len(refinement_rounds), 'g*', label='sb3-ppo')
    else:
        plt.plot(refinement_rounds, [baseline_val] * len(refinement_rounds), 'g-', label='RL baseline')

    plt.xlabel('Refinement Round')
    plt.ylabel('Reward')
    plt.legend()

    episodes_per_iter = experiment_1[1][0][0]["episodes_per_iteration"]
    plt.title(f'{env_name}: {len(refinement_rounds)} Iterations of {episodes_per_iter} Episodes')

    plt.show()

    # import tikzplotlib
    # tikzplotlib_fix_ncols(fig)
    # tikzplotlib.save("figures/lunar_lander_comparison_mean.tex")


def visualize_multiple_experiments2(experiments, env_name, method, baseline_val=None):
    refinement_rounds = list(range(1, len(experiments[0][1][0].keys()) + 1))

    fig = plt.figure()

    for exp_name, data in experiments:
        exp_1_plot_val_1, exp_1_plot_val_2 = get_iteration_averages(data, method)
        exp_1_plot_val_1 = np.array(list(exp_1_plot_val_1.values()))
        exp_1_plot_val_2 = np.array(list(exp_1_plot_val_2.values()))

        plt.plot(refinement_rounds, exp_1_plot_val_1, label=f'Mean Reward: {exp_name}')
        plt.fill_between(refinement_rounds, exp_1_plot_val_1 - exp_1_plot_val_2, exp_1_plot_val_1 + exp_1_plot_val_2
                         , alpha=0.2)

    if env_name == 'Lunar Lander':
        plt.plot(refinement_rounds, [136] * len(refinement_rounds), 'gd', label='sb3-dqn')
        plt.plot(refinement_rounds, [181] * len(refinement_rounds), 'gx', label='sb3-a2c')
        plt.plot(refinement_rounds, [223] * len(refinement_rounds), 'g*', label='sb3-ppo')
    else:
        plt.plot(refinement_rounds, [baseline_val] * len(refinement_rounds), 'g', label='Baseline Val')

    plt.xlabel('Refinement Round')
    plt.ylabel('Reward')
    plt.legend()

    episodes_per_iter = experiments[0][1][0][0]["episodes_per_iteration"]
    plt.title(f'{env_name}: {len(refinement_rounds)} Iterations of {episodes_per_iter} Episodes')

    plt.show()

    # import tikzplotlib
    # tikzplotlib_fix_ncols(fig)
    # tikzplotlib.save("figures/cartpole_multiple_comparison_mean.tex")


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


def load_all(files):
    return [load(l) for l in files]


if __name__ == '__main__':
    directory = "figures"
    if not os.path.exists(directory):
        os.makedirs(directory)

    cartpole_128_clusters = [
        f'pickles/results/final_exp{i}_CartPole-v1_num_traces_2500_powerTransformer_n_clusters_128_ri_15_ep_50.pk'
        for i in range(5)]
    cartpole_64_clusters = [
        f'pickles/results/cp_64_{i}_CartPole-v1_num_traces_2500_powerTransformer_n_clusters_64_ri_15_ep_50.pk'
        for i in range(5)]
    cartpole_32_clusters = [
        f'pickles/results/cp_test_{i}_CartPole-v1_num_traces_2500_powerTransformer_n_clusters_32_ri_15_ep_50.pk'
        for i in range(6)]

    mountain_car_256 = [
        f'pickles/results/A_exp{i}_MountainCar-v0_num_traces_2500_powerTransformer_n_clusters_256_ri_25_ep_50.pk'
        for i in [0, 1, 2, 3, 6]]
    mountain_car_128 = [
        f'pickles/results/mc_128_exp_{i}_MountainCar-v0_num_traces_2500_powerTransformer_n_clusters_128_ri_25_ep_50.pk'
        for i in range(5)
    ]
    mountain_car_64 = [
        f'pickles/results/mc_64_exp_{i}_MountainCar-v0_num_traces_2500_powerTransformer_n_clusters_64_ri_25_ep_50.pk'
        for i in range(5)]
    mountain_car_96 = [
        f'pickles/results/mc_96_exp_{i}_MountainCar-v0_num_traces_2500_powerTransformer_n_clusters_96_ri_25_ep_50.pk'
        for i in range(5)]

    acrobot_file_lda = [
        f'pickles/results/A_exp{i}_Acrobot-v1_num_traces_2500_powerTransformer_lda_2_n_clusters_256_ri_25_ep_50.pk'
        for i in range(10)]
    acrobot_file_manual = [
        f'pickles/results/A_exp{i}_Acrobot-v1_num_traces_2500_manualMapper_n_clusters_256_ri_25_ep_50.pk'
        for i in range(10)]
    lunar_lander_lda = [
        f'pickles/results/lda_mexp{i + 1}_LunarLander-v2_num_traces_2500_lda_powerTransformer_n_clusters_1024_ri_25_ep_50.pk'
        for i in range(5)]
    lunar_lander_manual = [
        f'pickles/results/mexp{i + 1}_LunarLander-v2_num_traces_2500_manualMapper_powerTransformer_n_clusters_1024_ri_25_ep_50.pk'
        for i in range(5)]

    baseline_values = {'MountainCar': -130, 'Acrobot': - 100, 'Cartpole': 200, 'LunarLander': 250}

    experiments = ['LunarLander', 'Acrobot', 'MountainCar', 'Cartpole']
    avg_method = 'mean_stddev'  # 'median_quantiles'

    # all_experiments = [load(l) for l in mountain_car_128]
    #
    # visualize_experiment_runs(all_experiments, 'Exp name', avg_method, )
    # # get_max_rew_and_std(all_experiments)
    #
    # visualize_multiple_experiments2([('CP 128', load_all(cartpole_128_clusters)), ('64', load_all(cartpole_64_clusters)), ('CP 32',load_all(cartpole_32_clusters))], 'Cartpole', avg_method, 200)
    # exit()

    for experiment in experiments:
        if experiment == 'LunarLander':
            visualize_multiple_experiments(('Manual Mapper', load_all(lunar_lander_manual)),
                                           ('LDA', load_all(lunar_lander_lda)),
                                           env_name='Lunar Lander', method=avg_method,
                                           baseline_val=baseline_values[experiment])
        if experiment == 'Acrobot':
            visualize_multiple_experiments(('Manual Mapper', load_all(acrobot_file_manual)),
                                           ('LDA', load_all(acrobot_file_lda)),
                                           env_name='Acrobot', method=avg_method, baseline_val=baseline_values[experiment])
        if experiment == 'MountainCar':
            visualize_multiple_experiments2([('k=256', load_all(mountain_car_256)),
                                             ('k=128', load_all(mountain_car_128)),
                                             ('k=64',load_all(mountain_car_64))], 'Mountain Car', avg_method, -130)
        if experiment == 'Cartpole':
            visualize_multiple_experiments2([('k=128', load_all(cartpole_128_clusters)),
                                             ('k=64', load_all(cartpole_64_clusters)),
                                             ('k=32',load_all(cartpole_32_clusters))], 'Cartpole', avg_method, 200)

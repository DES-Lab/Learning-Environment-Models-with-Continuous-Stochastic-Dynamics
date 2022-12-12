import pickle

import matplotlib.pylab as plt
import seaborn as sns


def plot_lunar_lander():
    with open('smc_lunar_lander.pickle', 'rb') as handle:
        data = pickle.load(handle)

    plot_options = {'goal': 0, 'crash': 1, 'time_out': 2, 'reward': 3}

    for agent in data.keys():
        agent_data = data[agent]
        policy_moves, random_moves = list(set(x[0] for x in agent_data.keys())), list(
            set([x[1] for x in agent_data.keys()]))

        policy_moves.sort()
        random_moves.sort()

        for data_type, index in plot_options.items():
            plot_data_array = []
            for pm in policy_moves:
                data_row = []
                for rm in random_moves:
                    data_row.insert(0, agent_data[(pm, rm)][index])  # goal_reached, crash, time_out, mean(rewards)
                plot_data_array.append(data_row)

            plot_data_array = list(map(list, zip(*plot_data_array)))

            y_label = random_moves.copy()
            y_label.reverse()

            fig = sns.heatmap(plot_data_array, xticklabels=policy_moves, yticklabels=y_label,
                              annot=True, fmt='g', cmap="Greens" if data_type in {'reward', 'goal'} else 'Reds')

            fig.set_xlabel('Policy Steps')
            fig.set_ylabel('Random Steps')
            title = f'LunarLander {agent} agent'
            fig.set_title(
                f'{title}: % {data_type} reached' if data_type != 'reward' else f'{title}: mean reward')
            plt.savefig(f'plots/lunar_lander_{agent}_{data_type}.pdf', dpi=300)
            plt.close()
            # plt.show()


def plot_cartpole_or_mountain_car(exp):
    assert exp in {'cartpole', 'mountain_car'}
    pickle_path = 'smc_cartpole.pickle' if exp == 'cartpole' else 'smc_mountain_car.pickle'

    with open(pickle_path, 'rb') as handle:
        data = pickle.load(handle)

    plot_options = {'reward': 3}

    for agent in data.keys():
        agent_data = data[agent]
        policy_moves, random_moves = list(set(x[0] for x in agent_data.keys())), list(
            set([x[1] for x in agent_data.keys()]))

        policy_moves.sort()
        random_moves.sort()

        for data_type, index in plot_options.items():
            plot_data_array = []
            for pm in policy_moves:
                data_row = []
                for rm in random_moves:
                    data_row.insert(0, agent_data[(pm, rm)][index])  # goal_reached, crash, time_out, mean(rewards)
                plot_data_array.append(data_row)

            plot_data_array = list(map(list, zip(*plot_data_array)))

            y_label = random_moves.copy()
            y_label.reverse()

            fig = sns.heatmap(plot_data_array, xticklabels=policy_moves, yticklabels=y_label,
                              annot=True, fmt='g', cmap="Greens" if data_type in {'reward', 'goal'} else 'Reds')

            fig.set_xlabel('Policy Steps')
            fig.set_ylabel('Random Steps')
            title = f'Cartpole {agent} agent' if exp == 'cartpole' else f'MountainCar {agent} agent'
            fig.set_title(
                f'{title}: % {data_type} reached' if data_type != 'reward' else f'{title}: mean reward')
            plt.savefig(f'plots/cartpole_{agent}_{data_type}.pdf', dpi=300)
            plt.close()


plot_cartpole_or_mountain_car(exp='mountain_car')

# plot_lunar_lander()
exit()

# plt.savefig('mqtt_heatmap.pdf', dpi=300)
# import tikzplotlib
# tikzplotlib.save("bluetooth_heatmap.tex")

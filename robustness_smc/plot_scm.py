import pickle

import matplotlib.pylab as plt
import seaborn as sns

plot_options = {'goal': 0, 'crash': 1, 'time_out': 2, 'reward': 3, 'episode_len': 4}


def plot_selected_categories(path_to_pickle, exp_name, categories):
    mutation_type = 'action' if 'action' in path_to_pickle else 'observation'

    for el in categories:
        assert el in {'reward', 'goal', 'crash', 'time_out', 'episode_len'}

    with open(path_to_pickle, 'rb') as handle:
        data = pickle.load(handle)

    for agent in data.keys():
        agent_data = data[agent]
        policy_moves, random_moves = list(set(x[0] for x in agent_data.keys())), list(
            set([x[1] for x in agent_data.keys()]))

        policy_moves.sort()
        random_moves.sort()

        for data_type in plot_categories:
            index = plot_options[data_type]

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
            title = f'{exp_name}, {agent}, {mutation_type}'
            fig.set_title(
                f'{title}: {data_type} %' if data_type not in {'reward', 'episode_len'} else f'{title}: mean {data_type}')
            plt.savefig(f'plots/{exp_name}_{agent}_{data_type}_mut_{mutation_type}.pdf', dpi=300)
            plt.close()


experiments = [('smc_mountain_car_action.pickle', 'MountainCar', ['reward']),
               ('smc_lunar_lander_action.pickle', 'LunarLander', ['reward', 'crash', 'goal', 'time_out']),
               ('smc_cartpole_action.pickle', 'CartPole', ['goal']),
               ('smc_bipedal_walker_action.pickle', 'BipedalWalker', ['reward', 'crash',]),
               ('smc_walker2d_action.pickle', 'Walker2d', ['reward', 'crash', 'goal', 'episode_len' ]), ] # episode_len

for pickle_file, exp_name, plot_categories in experiments:
    plot_selected_categories(pickle_file, exp_name, plot_categories)

# plot_lunar_lander()
exit()

# plt.savefig('mqtt_heatmap.pdf', dpi=300)
# import tikzplotlib
# tikzplotlib.save("bluetooth_heatmap.tex")

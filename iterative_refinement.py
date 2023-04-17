from statistics import mean

import numpy as np
from aalpy.learning_algs import run_JAlergia

from discretization_pipeline import get_observations_and_actions
from prism_scheduler import compute_weighted_clusters, ProbabilisticScheduler, PrismInterface
from utils import create_abstract_traces, remove_nan


class IterativeRefinement:
    def __init__(self, env, initial_model, abstract_traces, dim_reduction_pipeline, clustering_fun,
                 scheduler_type='probabilistic', count_observations=False):
        assert scheduler_type in {'probabilistic', 'deterministic'}
        self.env = env
        self.model = initial_model
        self.abstract_traces = abstract_traces
        self.dim_reduction_pipeline = dim_reduction_pipeline
        self.clustering_fun = clustering_fun
        self.scheduler_type = scheduler_type
        self.count_observations = count_observations

    def iteratively_refine_model(self, num_iterations, episodes_per_iteration, goal_state='succ'):

        nums_goal_reached = 0

        for refinement_iteration in range(num_iterations):

            # due to a bug in alergia
            remove_nan(self.model)
            # Make model input complete
            self.model.make_input_complete('sink_state')
            scheduler = PrismInterface(goal_state, self.model).scheduler
            if self.scheduler_type == 'probabilistic':
                scheduler = ProbabilisticScheduler(scheduler, truly_probabilistic=True)

            num_goal_reached_iteration = 0
            num_crashes_per_iteration = 0

            concrete_traces = []

            ep_rewards = []
            for _ in range(episodes_per_iteration):
                scheduler.reset()
                self.env.reset()
                ep_rew = 0
                ep_data = []

                # previous cluster and counter used for counting
                previous_cluster_count = None

                while True:
                    scheduler_input = scheduler.get_input()
                    if scheduler_input is None:
                        print('Could not schedule an action.')
                        break

                    action = np.array(int(scheduler_input[1:]))

                    observation, reward, done, _ = self.env.step(action)
                    ep_rew += reward
                    ep_data.append((observation.reshape(1, -1), action, reward, done))

                    abstract_obs = self.dim_reduction_pipeline.transform(np.array([observation]))

                    reached_cluster = self.clustering_fun.predict(abstract_obs)[0]
                    reached_cluster = f'c{reached_cluster}'

                    if self.count_observations:
                        if previous_cluster_count is None or (previous_cluster_count[0] != reached_cluster):
                            previous_cluster_count = (reached_cluster, 1)
                        else:
                            previous_cluster_count = (reached_cluster, previous_cluster_count[1] + 1)

                        reached_cluster = f'{previous_cluster_count[0]}_{previous_cluster_count[1]}'

                    if reward == '100':
                        nums_goal_reached += 1
                    elif reached_cluster == goal_state:
                        num_goal_reached_iteration += 1

                    if self.scheduler_type == 'probabilistic':
                        weighted_clusters = compute_weighted_clusters(abstract_obs, self.clustering_fun,
                                                                      len(set(self.clustering_fun.labels_)))

                        step_successful = scheduler.step_to(scheduler_input, weighted_clusters)
                    else:
                        step_successful = scheduler.step_to(scheduler_input, reached_cluster)
                        print(scheduler_input, reached_cluster)

                    if not step_successful:
                        print('Num steps:', len(ep_data))
                        print('Reached cluster:', reached_cluster)
                        print('Could not step in a model')
                        # if reached_cluster in scheduler.label_dict.keys():
                        #     scheduler.current_state = scheduler.label_dict[reached_cluster]

                        break

                    if done:
                        if reward == 100 and goal_state == 'GOAL':
                            nums_goal_reached += 1
                            print('Landed')
                        if reward == -100:
                            num_crashes_per_iteration += 1
                        break

                ep_rewards.append(ep_rew)
                print(f'Episode reward: {ep_rew}')
                concrete_traces.append(ep_data)

            nums_goal_reached += num_goal_reached_iteration
            print(f'# Goal Reached : {num_goal_reached_iteration} / {episodes_per_iteration}')
            print(f'# Crashes  : {num_crashes_per_iteration} / {episodes_per_iteration}')
            print(f'Mean ep. reward: {mean(ep_rewards)}')

            # refine model
            observation_space, action_space = get_observations_and_actions(concrete_traces)
            reduced_dim_obs_space = self.dim_reduction_pipeline.transform(observation_space)
            cluster_labels = self.clustering_fun.predict(reduced_dim_obs_space)

            iteration_abstract_traces = create_abstract_traces(concrete_traces, cluster_labels)

            self.abstract_traces.extend(iteration_abstract_traces)
            self.model = run_JAlergia(self.abstract_traces, automaton_type='mdp', path_to_jAlergia_jar='alergia.jar',
                                      optimize_for='accuracy')

            print(f'Refinement {refinement_iteration + 1} model size: {self.model.size} states')

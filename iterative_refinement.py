from statistics import mean

import numpy as np
from aalpy.learning_algs import run_JAlergia
from tqdm import tqdm

from discretization_pipeline import get_observations_and_actions
from prism_scheduler import compute_weighted_clusters, ProbabilisticScheduler, PrismInterface
from utils import remove_nan
from trace_abstraction import create_abstract_traces


class IterativeRefinement:
    def __init__(self, env, env_name, initial_model, abstract_traces, dim_reduction_pipeline, clustering_fun,
                 scheduler_type='probabilistic', count_observations=False):
        assert scheduler_type in {'probabilistic', 'deterministic'}
        self.env = env
        self.env_name = env_name
        self.model = initial_model
        self.abstract_traces = abstract_traces
        self.dim_reduction_pipeline = dim_reduction_pipeline
        self.clustering_fun = clustering_fun
        self.scheduler_type = scheduler_type
        self.count_observations = count_observations

    def iteratively_refine_model(self, num_iterations, episodes_per_iteration, goal_state='succ', early_stopping=0.8):

        nums_goal_reached = 0

        last_scheduler = None
        for refinement_iteration in range(num_iterations):

            # due to a bug in alergia
            remove_nan(self.model)
            # Make model input complete
            self.model.make_input_complete('sink_state')
            scheduler = PrismInterface(goal_state, self.model).scheduler
            if scheduler is None and last_scheduler:
                scheduler = last_scheduler
            elif scheduler is None and not last_scheduler:
                print('Initial scheduler could not be computed.')
                assert False

            last_scheduler = scheduler
            if self.scheduler_type == 'probabilistic':
                scheduler = ProbabilisticScheduler(scheduler, truly_probabilistic=True)

            num_goal_reached_iteration = 0
            num_crashes_per_iteration = 0

            concrete_traces = []

            ep_rewards = []
            for _ in tqdm(range(episodes_per_iteration)):
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

                    if self.dim_reduction_pipeline:
                        abstract_obs = self.dim_reduction_pipeline.transform(np.array([observation]))
                    else:
                        abstract_obs = [observation.reshape(1, -1)]

                    reached_cluster = self.clustering_fun.predict(abstract_obs)[0]
                    reached_cluster = f'c{reached_cluster}'

                    if self.count_observations:
                        if previous_cluster_count is None or (previous_cluster_count[0] != reached_cluster):
                            previous_cluster_count = (reached_cluster, 1)
                        else:
                            previous_cluster_count = (reached_cluster, previous_cluster_count[1] + 1)

                        reached_cluster = f'{previous_cluster_count[0]}_{previous_cluster_count[1]}'

                    elif reached_cluster == goal_state:
                        print('Target cluster reached.')
                        num_goal_reached_iteration += 1

                    if self.scheduler_type == 'probabilistic':
                        weighted_clusters = compute_weighted_clusters(scheduler, abstract_obs, scheduler_input,
                                                                      self.clustering_fun,
                                                                      len(set(self.clustering_fun.labels_)))

                        step_successful = scheduler.step_to(scheduler_input, weighted_clusters)
                    else:
                        step_successful = scheduler.step_to(scheduler_input, reached_cluster)
                        # print(scheduler_input, reached_cluster)

                    if not step_successful:
                        print('Num steps:', len(ep_data))
                        print('Reached cluster:', reached_cluster)
                        print('Could not step in a model')
                        # if reached_cluster in scheduler.label_dict.keys():
                        #     scheduler.current_state = scheduler.label_dict[reached_cluster]
                        break

                    if done:
                        if self.env_name == 'LunarLander-v2':
                            if reward >= 80:
                                print('Successfully landed.')
                                num_goal_reached_iteration += 1
                            else:
                                num_crashes_per_iteration += 1
                        elif self.env_name == 'MountainCar-v0' and len(ep_data) <= 200:
                            nums_goal_reached += 1
                        break

                ep_rewards.append(ep_rew)
                print(f'Episode reward: {ep_rew}')
                concrete_traces.append(ep_data)

            nums_goal_reached += num_goal_reached_iteration
            print(f'# Goal Reached : {num_goal_reached_iteration} / {episodes_per_iteration}')
            print(f'# Crashes  : {num_crashes_per_iteration} / {episodes_per_iteration}')
            print(f'Mean ep. reward: {mean(ep_rewards)} +- {np.std(ep_rewards, ddof=1)}')

            if num_goal_reached_iteration / episodes_per_iteration >= early_stopping:
                print(f"Stopping iteration loop as early stopping criterion (>= {early_stopping}) is met.")
                return

            # refine model
            observation_space, action_space = get_observations_and_actions(concrete_traces)
            if self.dim_reduction_pipeline:
                reduced_dim_obs_space = self.dim_reduction_pipeline.transform(observation_space)
            else:
                reduced_dim_obs_space = observation_space
            cluster_labels = self.clustering_fun.predict(reduced_dim_obs_space)

            iteration_abstract_traces = create_abstract_traces(self.env_name, concrete_traces, cluster_labels)

            self.abstract_traces.extend(iteration_abstract_traces)
            self.model = run_JAlergia(self.abstract_traces, automaton_type='mdp', path_to_jAlergia_jar='alergia.jar',
                                      heap_memory='-Xmx12G', optimize_for='accuracy')

            print(f'Refinement {refinement_iteration + 1} model size: {self.model.size} states')
            print('-' * 30)

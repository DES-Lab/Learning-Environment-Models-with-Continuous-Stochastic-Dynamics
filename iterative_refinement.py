import numpy as np
from aalpy.learning_algs import run_Alergia

from discretization_pipeline import get_observations_and_actions
from prism_scheduler import compute_weighted_clusters, ProbabilisticScheduler, PrismInterface
from utils import create_abstract_traces


class IterativeRefinement:
    def __init__(self, env, initial_model, traces, dim_reduction_pipeline, clustering_fun):
        self.env = env
        self.model = initial_model
        self.traces = traces
        self.dim_reduction_pipeline = dim_reduction_pipeline
        self.clustering_fun = clustering_fun

    def iteratively_refine_model(self, num_iterations, episodes_per_iteration, goal_state='GOAL'):

        nums_goal_reached = 0

        for refinement_iteration in range(num_iterations):
            # Make model input complete
            self.model.make_input_complete('sink_state')
            scheduler = PrismInterface(goal_state, self.model).scheduler
            scheduler = ProbabilisticScheduler(scheduler, truly_probabilistic=True)

            num_goal_reached_iteration = 0
            num_crashes_per_iteration = 0

            for _ in range(episodes_per_iteration):
                scheduler.reset()
                self.env.reset()
                ep_data = []

                while True:
                    scheduler_input = scheduler.get_input()
                    if scheduler_input is None:
                        print('Could not schedule an action.')
                        break

                    action = np.array(int(scheduler_input[1:]))

                    observation, reward, done, _ = self.env.step(action)

                    abstract_obs = self.dim_reduction_pipeline.transform(np.array([observation]))

                    reached_cluster = self.clustering_fun.predict(abstract_obs)[0]
                    reached_cluster = f'c{abstract_obs}'

                    if reached_cluster == goal_state:
                        num_goal_reached_iteration += 1

                    weighted_clusters = compute_weighted_clusters(abstract_obs, self.clustering_fun,
                                                                  len(set(self.clustering_fun.labels_)))

                    step_successful = scheduler.step_to(scheduler_input, weighted_clusters)
                    if not step_successful:
                        print('Could not step in a model')
                        break

                    ep_data.append((observation.reshape(1, -1), action, reward, done))

                    if done:
                        if reward == 100 and goal_state == 'GOAL':
                            nums_goal_reached += 1
                            print('Landed')
                        if reward == -100:
                            num_crashes_per_iteration += 1
                        break

                self.traces.append(ep_data)

            nums_goal_reached += num_goal_reached_iteration
            print(f'# Goal Reached : {num_goal_reached_iteration} / {episodes_per_iteration}')
            print(f'# Crashes  : {num_crashes_per_iteration} / {episodes_per_iteration}')

            # refine model
            observation_space, action_space = get_observations_and_actions(self.traces)
            reduced_dim_obs_space = self.dim_reduction_pipeline.transform(observation_space)
            cluster_labels = self.clustering_fun.predict(reduced_dim_obs_space)

            extended_data = create_abstract_traces(self.traces, cluster_labels)
            self.model = run_Alergia(extended_data, automaton_type='mdp')

            print(f'Refinement {refinement_iteration + 1} model size: {self.model.size} states')

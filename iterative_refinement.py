from collections import defaultdict
from statistics import mean, stdev

import numpy as np
from aalpy.learning_algs import run_JAlergia

from discretization_pipeline import get_observations_and_actions
from prism_scheduler import compute_weighted_clusters, ProbabilisticScheduler, PrismInterface
from trace_abstraction import create_abstract_traces
from utils import remove_nan, CARTPOLE_CUTOFF, ACROBOT_GOAL, MOUNTAIN_CAR_GOAL, mdp_to_state_setup, save


class IterativeRefinement:
    def __init__(self, env, env_name, initial_model, abstract_traces, dim_reduction_pipeline, clustering_fun,
                 scheduler_type='probabilistic', count_observations=False, num_agent_steps=None, agent=None):
        assert scheduler_type in {'probabilistic', 'deterministic'}
        self.env = env
        self.env_name = env_name
        self.model = initial_model
        self.abstract_traces = abstract_traces
        self.dim_reduction_pipeline = dim_reduction_pipeline
        self.clustering_fun = clustering_fun
        self.scheduler_type = scheduler_type
        self.count_observations = count_observations

        self.num_agent_steps = num_agent_steps
        self.agent = agent

        self.exp_name = f'{dim_reduction_pipeline.pipeline_name}' \
                        f'_n_clusters_{len(set(self.clustering_fun.labels_))}'

    def iteratively_refine_model(self, num_iterations, episodes_per_iteration, goal_state='succ', early_stopping=0.8):

        results = defaultdict(dict)

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
            for episode_index in range(episodes_per_iteration):
                scheduler.reset()
                obs = self.env.reset()
                ep_rew = 0

                if self.num_agent_steps:
                    for _ in range(int(self.num_agent_steps)):
                        action, _ = self.agent.predict(obs)
                        obs, rew, _, _ = self.env.step(action)
                        ep_rew += rew
                ep_data = []

                # previous cluster and counter used for counting
                previous_cluster_count = None

                num_steps_model = 0
                add_info = ''

                while True:
                    scheduler_input = scheduler.get_input()
                    if scheduler_input is None:
                        print('Could not schedule an action.')
                        break

                    action = np.array(int(scheduler_input[1:]))
                    num_steps_model += 1
                    observation, reward, done, _ = self.env.step(action)
                    # self.env.render()

                    # add cutoff for Cartpole
                    if "CartPole" in self.env_name and len(ep_data) >= CARTPOLE_CUTOFF:
                        done = True

                    ep_rew += reward
                    ep_data.append((observation.reshape(1, -1), action, reward, done))

                    if self.dim_reduction_pipeline is not None:
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
                        ep_rew += -5000
                        print('Num steps:', len(ep_data))
                        print('Reached cluster:', reached_cluster)
                        print('Could not step in a model')
                        break

                    if done:
                        if self.env_name == 'LunarLander-v2':
                            if reward >= 80:
                                add_info += 'Successfully landed'
                                num_goal_reached_iteration += 1
                            else:
                                num_crashes_per_iteration += 1
                        elif self.env_name == 'MountainCar-v0' and len(ep_data) <= MOUNTAIN_CAR_GOAL:
                            num_goal_reached_iteration += 1
                            add_info += f'Goal Reached under {MOUNTAIN_CAR_GOAL} steps'
                        elif "Acrobot" in self.env_name and len(ep_data) <= ACROBOT_GOAL:
                            nums_goal_reached += 1
                            add_info += f'Goal Reached under {ACROBOT_GOAL} steps'
                        break

                ep_rewards.append(ep_rew)
                print(
                    f'Episode {episode_index}/{episodes_per_iteration} reward: {ep_rew}, Model Steps {num_steps_model} {add_info}')
                concrete_traces.append(ep_data)

            nums_goal_reached += num_goal_reached_iteration
            print(f'# Goal Reached : {num_goal_reached_iteration} / {episodes_per_iteration}')
            print(f'# Crashes  : {num_crashes_per_iteration} / {episodes_per_iteration}')
            print(f'Mean ep. reward: {mean(ep_rewards)} +- {stdev(ep_rewards)}')

            if num_goal_reached_iteration / episodes_per_iteration >= early_stopping:
                print(f"Stopping iteration loop as early stopping criterion (>= {early_stopping}) is met.")
                return

            # refine model
            observation_space, action_space = get_observations_and_actions(concrete_traces)
            if self.dim_reduction_pipeline is not None:
                reduced_dim_obs_space = self.dim_reduction_pipeline.transform(observation_space)
            else:
                reduced_dim_obs_space = observation_space
            cluster_labels = self.clustering_fun.predict(reduced_dim_obs_space)

            iteration_abstract_traces = create_abstract_traces(self.env_name, concrete_traces, cluster_labels)

            self.abstract_traces.extend(iteration_abstract_traces)
            self.model = run_JAlergia(self.abstract_traces, automaton_type='mdp', path_to_jAlergia_jar='alergia.jar',
                                      heap_memory='-Xmx12G', optimize_for='accuracy')

            print(f'Refinement {refinement_iteration + 1} model size: {self.model.size} states')

            # save results
            ep_lens = [len(e) for e in concrete_traces]
            results[refinement_iteration]['reward'] = mean(ep_rewards), stdev(ep_rewards)
            results[refinement_iteration]['model_size'] = self.model.size
            results[refinement_iteration]['goal_reached'] = num_goal_reached_iteration
            results[refinement_iteration]['goal_reached_percentage'] = num_goal_reached_iteration / episodes_per_iteration
            results[refinement_iteration]['crash'] = num_crashes_per_iteration
            results[refinement_iteration]['episode_len'] = mean(ep_lens), stdev(ep_lens)
            results[refinement_iteration]['model'] = mdp_to_state_setup(self.model)
            results[refinement_iteration]['iteration_episodes'] = episodes_per_iteration

            print('-' * 45)

            save(results, f'pickles/results/{self.exp_name}.pk')

        return results

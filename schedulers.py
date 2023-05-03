import os
import random
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean

import aalpy.paths
import numpy as np
from aalpy.utils import mdp_2_prism_format
from scipy.stats import norm


class Scheduler:
    def __init__(self, initial_state, transition_dict, label_dict, scheduler_dict):
        self.scheduler_dict = scheduler_dict
        self.initial_state = initial_state
        self.transition_dict = transition_dict
        self.label_dict = label_dict
        self.current_state = None

    def get_input(self):
        if self.current_state is None:
            print("Return none because current state is none")
            return None
        else:
            # print("Current state is not none")
            if self.current_state not in self.scheduler_dict:
                return None
            return self.scheduler_dict[self.current_state]

    def reset(self):
        self.current_state = self.initial_state

    def poss_step_to(self, input):
        output_labels = []
        trans_from_current = self.transition_dict[self.current_state]
        found_state = False
        for (prob, action, target_state) in trans_from_current:
            if action == input:
                output_labels.extend(self.label_dict[target_state])
        return output_labels

    def step_to(self, input, output):
        reached_state = None
        trans_from_current = self.transition_dict[self.current_state]
        found_state = False
        for (prob, action, target_state) in trans_from_current:
            if action == input and output in self.label_dict[target_state]:
                reached_state = self.current_state = target_state
                found_state = True
                break
        if not found_state:
            reached_state = None

        return reached_state

    def get_available_actions(self):
        trans_from_current = self.transition_dict[self.current_state]
        return list(set([action for prob, action, target_state in trans_from_current]))


class ProbabilisticScheduler:
    def __init__(self, scheduler: Scheduler, truly_probabilistic, max_state_size=4):
        self.scheduler_dict = scheduler.scheduler_dict
        self.initial_state = scheduler.initial_state
        self.transition_dict = scheduler.transition_dict
        self.label_dict = scheduler.label_dict
        self.current_state = None
        self.truly_probabilistic = truly_probabilistic
        self.max_state_size = max_state_size
        self.discount = 0.5

    def get_input_preferences(self):
        input_preference = defaultdict(float)
        # print(self.current_state)
        for (s, certainty) in self.current_state:
            if s in self.scheduler_dict:
                input_preference[self.scheduler_dict[s]] += certainty
        # print(input_preference)
        if len(input_preference) == 0:
            return None
        elif len(input_preference) == 1:
            return input_preference
        else:
            return input_preference

    def get_input(self):
        input_preference = self.get_input_preferences()
        if not input_preference:
            return None
        pref_list = list(input_preference.items())
        inputs, weights = zip(*pref_list)
        if self.truly_probabilistic:
            return random.choices(inputs, weights=weights, k=1)[0]
        else:
            return inputs[np.argmax(weights)]

    def reset(self):
        self.current_state = [(self.initial_state, 1.0)]

    def poss_step_to(self, action):
        labels = set()
        for (state, certainty) in self.current_state:
            trans_from_current = self.transition_dict[state]
            for (prob, action_loop, target_state) in trans_from_current:
                if action_loop == action:
                    cluster_label = list(filter(lambda l: l.startswith("c"), self.label_dict[target_state]))
                    if len(cluster_label) > 0:
                        labels.add(cluster_label[0])
        return list(labels)

    def _poss_step_to(self, state, action):
        labeled_states = []
        trans_from_current = self.transition_dict[state]
        for (prob, action_loop, target_state) in trans_from_current:
            if action_loop == action:
                labeled_states.append((target_state, self.label_dict[target_state], prob))
        return labeled_states

    def has_transition(self, action, obs):
        labeled_states = []
        for (s, cert) in self.current_state:
            trans_from_current = self.transition_dict[s]
            for (prob, action_loop, target_state) in trans_from_current:
                if action_loop == action and obs in self.label_dict[target_state]:
                    return True
        return False

    def step_to(self, action, weighted_clusters: dict):
        # assume weighted_clusters : dict : cluster_label -> probabilities, where probability is large enough
        successors = []
        for s, certainty in self.current_state:
            current_label = self.label_dict[s]
            possible_successors = self._poss_step_to(s, action)
            for succ, labels, prob in possible_successors:
                for cluster in weighted_clusters.keys():
                    if cluster in labels:
                        cluster_weight = weighted_clusters[cluster]  # math.exp(-weighted_clusters[cluster])
                        successors.append((succ, certainty * cluster_weight))

        successors.sort(key=lambda x: x[1], reverse=True)

        cert_sum = sum([v[1] for v in successors])
        if len(successors) == 0 or cert_sum == 0:
            return False
        if len(successors) > self.max_state_size:
            successors = successors[0:self.max_state_size]
        # normalize certainty to one
        combined_state = defaultdict(float)
        for s, cert in successors:
            combined_state[s] += cert / cert_sum
        self.current_state = list(combined_state.items())
        return True


class PrismInterface:
    def __init__(self, destination, model, num_steps=None, maximize=True):
        self.tmp_dir = Path("tmp_prism")
        self.destination = destination
        self.model = model
        self.num_steps = num_steps
        self.maximize = maximize
        if type(destination) != list:
            destination = [destination]
        destination = "_or_".join(destination)
        self.tmp_mdp_file = (self.tmp_dir / f"po_rl_{destination}.prism")
        self.current_state = None
        self.tmp_dir.mkdir(exist_ok=True)
        self.prism_property = self.create_mc_query()
        mdp_2_prism_format(self.model, "porl", output_path=self.tmp_mdp_file)

        self.adv_file_name = (self.tmp_dir.absolute() / f"sched_{destination}.adv")
        self.concrete_model_name = str(self.tmp_dir.absolute() / f"concrete_model_{destination}")
        self.property_val = 0
        self.call_prism()
        if os.path.exists(self.adv_file_name):
            self.parser = PrismSchedulerParser(self.adv_file_name, self.concrete_model_name + ".lab",
                                               self.concrete_model_name + ".tra")
            self.scheduler = Scheduler(self.parser.initial_state, self.parser.transition_dict,
                                       self.parser.label_dict, self.parser.scheduler_dict)
            os.remove(self.tmp_mdp_file)
            os.remove(self.adv_file_name)
            os.remove(self.concrete_model_name + ".lab")
            os.remove(self.concrete_model_name + ".tra")
        else:
            self.scheduler = None

    def create_mc_query(self):
        if type(self.destination) != list:
            destination = [self.destination]
        else:
            destination = self.destination
        destination = "|".join(map(lambda d: f"\"{d}\"", destination))
        opt_string = "Pmax" if self.maximize else "Pmin"
        prop = f"{opt_string}=?[F {destination}]" if not self.num_steps else \
            f'{opt_string}=?[F<{self.num_steps} {destination}]'
        return prop

    def call_prism(self):
        import subprocess
        from os import path

        self.property_val = 0

        destination_in_model = False
        for s in self.model.states:
            if self.destination in s.output.split("__"):
                destination_in_model = True
                break

        prism_file = aalpy.paths.path_to_prism.split('/')[-1]
        path_to_prism_file = aalpy.paths.path_to_prism[:-len(prism_file)]
        file_abs_path = path.abspath(self.tmp_mdp_file)
        proc = subprocess.Popen(
            [aalpy.paths.path_to_prism, file_abs_path, "-pf", self.prism_property, "-noprob1", "-exportadvmdp",
             self.adv_file_name, "-exportmodel", f"{self.concrete_model_name}.all"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=path_to_prism_file)
        out = proc.communicate()[0]
        out = out.decode('utf-8').splitlines()
        for line in out:
            # print(line)
            if not line:
                continue
            if 'Syntax error' in line:
                print(line)
            else:
                if "Result:" in line:
                    end_index = len(line) if "(" not in line else line.index("(") - 1
                    try:
                        self.property_val = float(line[len("Result: "): end_index])
                    except:
                        print("Result parsing error")
        proc.kill()
        return self.property_val


class PrismSchedulerParser:
    def __init__(self, scheduler_file, label_file, transition_file):
        with open(scheduler_file, "r") as f:
            self.scheduler_file_content = f.readlines()
        with open(label_file, "r") as f:
            self.label_file_content = f.readlines()
        with open(transition_file, "r") as f:
            self.transition_file_content = f.readlines()
        self.label_dict = self.create_labels()
        self.transition_dict = self.create_transitions()
        self.scheduler_dict = self.parse_scheduler()
        self.initial_state = next(filter(lambda e: "init" in e[1], self.label_dict.items()))[0]
        self.actions = set()
        for l in self.transition_dict.values():
            for _, action, _ in l:
                self.actions.add(action)
        self.actions = list(self.actions)

    def create_labels(self):
        label_dict = dict()
        header_line = self.label_file_content[0]
        label_lines = self.label_file_content[1:]
        header_dict = dict()
        split_header = header_line.split(" ")
        for s in split_header:
            label_id = s.strip().split("=")[0]
            label_name = s.strip().split("=")[1].replace('"', '')
            header_dict[label_id] = label_name
        for l in label_lines:
            state_id = int(l.split(":")[0])
            label_ids = l.split(":")[1].split(" ")
            label_names = set(
                map(lambda l_id: header_dict[l_id.strip()], filter(lambda l_id: l_id.strip(), label_ids)))
            label_dict[state_id] = label_names
        return label_dict

    def create_transitions(self):
        header_line = self.transition_file_content[0]
        transition_lines = self.transition_file_content[1:]
        transitions = defaultdict(list)
        for t in transition_lines:
            split_line = t.split(" ")
            source_state = int(split_line[0])
            target_state = int(split_line[2])
            prob = float(split_line[3])
            action = split_line[4].strip()
            transitions[source_state].append((prob, action, target_state))
        return transitions

    def parse_scheduler(self):
        header_line = self.scheduler_file_content[0]
        transition_lines = self.scheduler_file_content[1:]
        scheduler = dict()
        for t in transition_lines:
            split_line = t.split(" ")
            source_state = int(split_line[0])
            action = split_line[4].strip()
            if source_state in scheduler:
                assert action == scheduler[source_state]
            else:
                scheduler[source_state] = action
        return scheduler


def compute_weighted_clusters(scheduler, conc_obs, action, clustering_function, nr_outputs):
    cluster_distances = clustering_function.transform(conc_obs).tolist()[0]

    reachable_clusters = set()
    for (s, certainty) in scheduler.current_state:
        possible_successors = scheduler._poss_step_to(s, action)
        for (succ, labels, prob) in possible_successors:
            for l in labels:
                if l.startswith('c'):
                    reachable_clusters.add(l)
    if len(reachable_clusters) > 1:
        cluster_distances = sorted(
            [(f"c{ind_c[0]}", ind_c[1]) for ind_c in enumerate(cluster_distances) if
             f"c{ind_c[0]}" in reachable_clusters],
            key=lambda x: x[1])
    else:
        if len(reachable_clusters) == 0:
            return dict()
        else:
            return {list(reachable_clusters)[0]: 1}

    nr_clusters = len(cluster_distances)
    avg_distance = mean([ind_c[1] for ind_c in cluster_distances])
    variance = (sum([(ind_c[1] - avg_distance) ** 2 for ind_c in cluster_distances]) / nr_clusters)
    cluster_distances = cluster_distances[0:nr_outputs]
    weighted_clusters = dict([(v[0],
                               1 - norm.cdf(v[1], loc=avg_distance,
                                            scale=sqrt(variance)
                                            )) for v in cluster_distances])

    return weighted_clusters

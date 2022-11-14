import random
from collections import defaultdict

from aalpy.SULs import MdpSUL
from aalpy.automata import Mdp, MdpState
from aalpy.learning_algs import run_Alergia

from ansamble_creation import get_trace_suffixes
from utils import compress_trace


def get_small_pomdp():
    q0 = MdpState("q0", "init")
    q1 = MdpState("q1", "beep")
    q2 = MdpState("q2", "beep")
    q3 = MdpState("q3", "coffee")
    q4 = MdpState("q4", "tea")

    q0.transitions['but'].append((q0, 1))
    q0.transitions['coin'].append((q1, 0.8))
    q0.transitions['coin'].append((q2, 0.2))

    q1.transitions['coin'].append((q1, 1))
    q1.transitions['but'].append((q3, 1))

    q2.transitions['coin'].append((q2, 0.3))
    q2.transitions['coin'].append((q1, 0.7))
    q2.transitions['but'].append((q4, 1))

    q3.transitions['coin'].append((q3, 1))
    q3.transitions['but'].append((q3, 1))

    q4.transitions['coin'].append((q4, 1))
    q4.transitions['but'].append((q4, 1))

    return Mdp(q0, [q0, q1, q2, q3, q4])


def compute_assemble_mdp(alergia_traces):
    cluster_traces = defaultdict(list)
    assemble_mdps = dict()

    for trace in alergia_traces:
        trace_suffixes = get_trace_suffixes(trace)
        for suffix in trace_suffixes:
            output = suffix[0][1]
            if len(suffix) > 1:
                cluster_traces[output].append([output, ] + suffix[1:])

    for output, cluster_traces in cluster_traces.items():
        mdp = run_Alergia(cluster_traces, 'mdp')
        assemble_mdps[output] = mdp

    return assemble_mdps


pomdp = get_small_pomdp()
input_al = pomdp.get_input_alphabet()
sul = MdpSUL(pomdp)

data = []
for _ in range(10000):
    sample = []
    sul.pre()
    for _ in range(15):
        i = random.choice(input_al)
        o = sul.step(i)
        sample.append((i, o))
    data.append(sample)

model_map = compute_assemble_mdp(data)
for k, v in model_map.items():
    v.visualize(k)

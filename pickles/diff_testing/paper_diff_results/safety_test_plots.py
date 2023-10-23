import glob
import pickle
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


exp_name = "LunarLander"

if exp_name == "LunarLander":
    pass_string = "Landed"
    fail_string = "Crash"

elif exp_name == "CartPole":
    pass_string = "Pass"
    fail_string = "Fail"
# collect relevant files
all_results = dict()
max_nr_tests = 200
policies = None

cluster_importance_file = f"cluster_importance_{exp_name}.txt"
sorted_clusters = []
with open(cluster_importance_file) as cluster_imp_handle:
    lines = cluster_imp_handle.readlines()
    for line in lines:
        start = line.index("('")
        end = line.index("',")
        cluster_name = line[start + 2:end]
        sorted_clusters.append(cluster_name)

for result_file in glob.glob(f'./diff_test_{exp_name}*.pickle'):
    if "time" in result_file:
        continue
    cluster = re.findall(r'c\d+\.pickle', result_file)[0]
    cluster = cluster.replace('.pickle', '')
    print(cluster)
    with open(result_file, 'rb') as fp:
        results = pickle.load(fp)
    print(results)
    all_results[cluster] = results
    if policies is None:
        # to keep ordering fixed
        policies = list(results.keys())

i = 0
normalized_results = defaultdict(list)
inconclusive_clusters = dict()
cluster_names = []
tested_cluster_names = []
for cluster, results in all_results.items():
    tested_cluster_names.append(cluster)
sorted_tested_clusters = []

for i, c in enumerate(sorted_clusters):
    if c in tested_cluster_names:
        sorted_tested_clusters.append((c, i))

for (c, i) in sorted_tested_clusters:
    cluster_names.append(c)

for cluster in cluster_names:
    # for cluster, results in all_results.items():
    results = all_results[cluster]  # create explicitly to avoid ordering issues
    inconc_added = False
    for policy_name in policies:
        result_counter = results[policy_name]
        nr_tests = sum(result_counter.values())
        if inconc_added == False:
            if nr_tests >= max_nr_tests:
                inconclusive_clusters[cluster] = True
            else:
                inconclusive_clusters[cluster] = False
            inconc_added = True
        normalized_pass = result_counter[pass_string] / nr_tests
        normalized_fail = result_counter[fail_string]
        if exp_name == "LunarLander":
            normalized_fail += result_counter["Time_out"]
        normalized_fail = normalized_fail / nr_tests
        normalized_results[policy_name].append(normalized_fail)

width = 0.4  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

# print(inconclusive_clusters)
# print(normalized_results)
# exit(0)
once = False
diff_labels = ["" if inconclusive_clusters[c] else "diff" for c in cluster_names]
x = np.array([i for i, c in enumerate(cluster_names) if not inconclusive_clusters[c]])
print(x)
colors = ["#fc0303", "#0303fc"]
for policy, measurement in normalized_results.items():
    offset = width * multiplier
    y = np.array([measurement[i] for i, c in enumerate(cluster_names) if not inconclusive_clusters[c]])
    rects = ax.bar(x + offset + width / 2, y, width, color=colors[multiplier], label=policy)
    print(rects)
    # ax.bar_label(rects, padding=3,fmt='%1.3f',rotation='vertical')
    multiplier += 1

x = np.array([i for i, c in enumerate(cluster_names) if inconclusive_clusters[c]])
print(x)
multiplier = 0
for policy, measurement in normalized_results.items():
    offset = (width * multiplier)
    y = np.array([measurement[i] for i, c in enumerate(cluster_names) if inconclusive_clusters[c]])
    rects = ax.bar(x + offset + width / 2, y, width, color=colors[multiplier], alpha=0.3)
    # print(rects)
    # ax.bar_label(rects, padding=3,fmt='%1.3f',rotation='vertical')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Fail Ratio')
ax.set_title('Differential Safety Test Results')
x = np.arange(len(cluster_names))
ax.set_xticks(x + width, cluster_names)
ax.legend(loc='upper left', ncols=3)
ylim = 0.56 if exp_name == "LunarLander" else 1.15
ax.set_ylim(0, ylim)

plt.show()

# tikzplotlib_fix_ncols(fig)
#
# import tikzplotlib
# tikzplotlib.save('bar_plot_diff_testing_lunar_lander.tex')
# # plt.savefig(f'diff_testing_{exp_name}.png')

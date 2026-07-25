![](https://github.com/DES-Lab/Learning-Environment-Models-with-Continuous-Stochastic-Dynamics/blob/master/figures/castle_intro.PNG)

# Learning Environment Models with Continuous Stochastic Dynamics - with an Application to Deep RL Testing 
This repository contains all code required to reproduce experiments reported in "Learning Environment Models with Continuous Stochastic Dynamics" paper.

**NOTE**: master branch contains code with updated dependencies. To see exact paper submission code, go to its [dedicated branch](https://github.com/DES-Lab/Learning-Environment-Models-with-Continuous-Stochastic-Dynamics/tree/paper-submission).

## Reproducibility and Setup
For the computation of schedulers install [Prism Model Checker](https://www.prismmodelchecker.org/).
Prism requires Java; model learning itself does not, as it uses AALpy's pure-Python Alergia.

To reproduce all experiments, we recommend that you create a new Python virtual enviroment in which you can install all recquirements.
Code has been tested with Python 3.9/3.12 and Prism 4.7.
```
python -m venv myenv
source myenv/bin/activate // Linux and Mac
myenv\Scripts\activate.bat // Windows
python -m pip install --upgrade pip // update pip
```
To install requirements:
```
pip install -r requirements.txt
```

### Environments
The code runs on [gymnasium](https://gymnasium.farama.org/). Gymnasium retired the `LunarLander-v2`
id in favour of `LunarLander-v3` (same environment, a few bug fixes), so `utils.make_env` translates
the id when the environment is created. Experiments keep referring to it as `LunarLander-v2` so that
the pipelines, clusterings and results pickled with the earlier code remain usable.

## Code structure
Main files:
- main.py - an example file which can be used to learn enviromental models of well established RL benchmarks. Example on how to use our approach
- iterative_refinement.py - file which contains code which iteratively refines learned model with respect to some goal
- diff_testing.py - All code required to differentially test multiple RL agents
 
Util files:
- trace_abstraction.py - convert high dimensional sequances to their abstract/discrete representations
- discretization_pipeline.py - helper file with methods used to reduce data dimensions
- schedulers.py - interface between learned MDPs and PRISM
- visualization_util.py - visualize results of experiments
- utils.py - minor utility functions

## Learning Environment Models with Continuous Stochastic Dynamics
To reproduce an experiment, simply call `experiment_cmd_runner.py` with appropriate arguments, as shown in the following line:
```
python experiment_cmd_runner.py --path_to_prism "C:/Program Files/prism-4.7/bin/prism.bat" --env_name Acrobot --dim_reduction manual --num_initial_traces 2500 --num_clusters 256 --num_iterations 10 --episodes_in_iter 10 --exp_prefix exp_1_ --seed 101
```
Alternatively, you can change variable values in `main.py` and execute any experiment from there. 

To set a constant random seed for reproducibility, simply define an --seed argument or set the seed in the 'main.py' file.

To visualize the plots found in the paper, run `visualization_util.py`. Visualization_util can also be used to visualize new runs.

### Output structure
Outputs of iterative refinements will be printed to console as the algorithm progresses, and after every refinement iteration multiple values will be saved to a pickle, so that experiments can be reproduced afterwards. 
For more details, check the bottom of `iteratively_refine_model` function in `iterative_refinement.py`.

## Differential Testing
All code required to differentially test 2 agents with CASTLE is found in `diff_testing.py`.
When running the script, please replace `path_to_prism` in line 26 with appropriate install path.

To run differential testing with learned models (and during differential testing fined tuned), simply run
`diff_testing.py`. To switch between LunarLander and Cartpole experiments, change the value of experiment variable in line 313.

For each cluster of interest, results of differential testing will be saved to a pickle and .txt file found in
`pickles/diff_testing/` folder.

Differential testing plots found in the paper can be visualized with `pickles/diff_testing/paper_diff_results/saftey_test_plots.py`.
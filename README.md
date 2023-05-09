# Learning Environment Models with Continuous Stochastic Dynamics 
This repository contains all code required to reproduce experiments reported in "Learning Environment Models with Continuous Stochastic Dynamics" paper.

## Reproducibility and Setup
For the computation of schedulers install [Prism Model Checker](https://www.prismmodelchecker.org/).
We have used Java12(openjdk 12.0.2) run alergia.jar and Prism.

To reproduce all experiments, we recommend that you create a new Python virtual enviroment in which you can install all recquirements.
Code has been tested with Python 3.9 and Prism 4.7.
```
python -m venv myenv
source myenv/bin/activate // Linux and Mac
myenv\Scripts\activate.bat // Windows
```
To install requirements:
```
pip install -r requirements.txt
```

To reproduce an experiment, simply call `experiment_cmd_runner.py` with appropriate arguments, as shown in the following line:
```
python experiment_cmd_runner.py --path_to_prism "C:/Program Files/prism-4.7/bin/prism.bat" --path_to_alergia alergia.jar --env_name Acrobot --dim_reduction manual --num_initial_traces 2500 --num_clusters 256 --num_iterations 10 --episodes_in_iter 10 --exp_prefix exp_1_ --seed 101
```
Alternatively, you can change variable values in `main.py` and execute any experiment from there. 

To set a constant random seed for reproducibility, simply define an --seed argument or set the seed in the 'main.py' file.

To visualize the results, parameterize and use visualization_util.py.

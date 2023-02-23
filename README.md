# NetlogoDataAnalysis

Repository tracking Python source code used for analysing NetLogo simulation results for the paper "Learning stigmergic communication in multi-agent systems".

# Structure

Scripts:

 * the `auto_exploration.py` script automatically produces the paper plots given the experiment `.csv` data file based on some configuration params (see "Usage" section below)
 * the `manual_exploration.py` script contains un-optimised code initially used to perform a first exploratory analysis of the generated datasets, that could still be useful to produce additional and different kinds of plots (the `auto_exploration.py` script is actually a refactoring and refinement of this rough codebase)

Data:

 * `data/RL-slimes/behavioural` contains several folders with the generated data of all the different experiments that have been done during this research, roughly organised in "categories" of experiments (e.g. using different observation spaces, different action spaces, etc.)
 * `.csv` files contain the generated data of a given experiment (identified univocally by the filename). The first row has column headers, which are mostly self-explanatory and also described in the paper
 * `.params` files contain the simulation and learning parameters of a given experiment (identified univocally by the same filename used for the corresponding `.csv` file, so as to be easily able to connect generated data to its experimental setting, and reproduce the experiments)

# Installation

Just clone the repo, install the required packages, and you're good to go :)
Requirements are Python 3.9+ with PIP.

# Usage

TBD

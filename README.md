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
Requirements are Python 3.9+ with PiP.

# Usage

Given an input directory storing either raw files produced by the NetLogo simulations [1], or the files already split into `.params` and `.csv` parts, the script `auto_exploration.py` proceeds in 3 stages, applied to each experiment (= `.csv` file):

 1. the configured performance plots are produced
 2. both global and per-agent actions distributions are computed, if not skipped or already done
 3. both global and per-agent actions distributions are plotted, if not skipped or already done

That's it, really: just put the `.txt` files produced from simulations in the right input folder (see below) and launch the script :)
(An informative log is produced on standard output)

### Main parameters

All 3 steps can be configured as follows:

 1. the list of which plots to produce can be configured via

   * variable `whats` = a list of the `.csv` columns (y axis) to plot one at a time against the number of episodes (x axis). The number of plots will be equal to `len(whats)`
   * variable `whatlables`= the corresponding list of lables to print on the title of the plots
   * variable `ylables` = the corresponding list of lables to print on the y axis of the plots

 2. the actions distributions to compute  via

   * variable `skip_actions` = `True` to skip computation (hence plotting)
   * variable `action_space` = list of actions to consider

 3. how to plot such actions distributions via

   * `agent_mca_look` = a dictionary mapping the action lable in `aspace_lables` (key, see below) to its color and marker in a "look-and-feel" dictionary shaped as below (exemplary)

```
{
  'marker': 's',
  'color': '#648FFF'
}
```
   * `aspace_labels` = the corresponding list of labels to print on the plot (1 per action in `action_space`)

### Folders

The directories that the script uses are configurable as follows:

 * `root_in` = input folder whre raw datafiles produced by NetLogo simulations [1] (`.txt`) are expected to be
 * `root_out` = output folder where the above `.txt` files are split into data (`.csv`) and parameters (`.params`) files
 * `plots_root` = output folder where configured plots are put

If no files are found in `root_in` the scripts then expects files to be already split up in folder `root_out`.

### Other parameters

Other miscellaneous configuration parameters are:

 * `header_size` = the number of lines constituting the `.params` file
 * `alable` = how to call the simulation agents (e.g. learners or non-learners) in plots
 * `n_agents` = the number of agents in the simulation
 * `batch_size` = how to split agents in groups for plotting (TL;DR: 50 agents on a single actions distribution plot are too crowded, whereas a batch of 10 agents each in 5 plots -- for 50 total agents -- are much more manageable)

# KILL-experiments (WOA 2022)
KILL - Knowledge Injection via Lambda Layer experiments (WOA 2022).

## Users

Users can replicate the results of KILL algorithm applied to the 
poker hand dataset available at https://archive.ics.uci.edu/ml/datasets/Poker+Hand.

### Requirements

- python 3.9
- java 11 (for Antlr4 support)
- antlr4-python3-runtime 4.9.3
- tensorflow 2.7.0
- scikit-learn 1.0.2
- pandas 1.4.2
- numpy 1.22.3
- 2ppy 0.4.0
- psyki 0.1.18
- scipy 1.8.0
- setuptools 62.1.0
- matplotlib 3.5.1

### Setup

You can execute a list of predefined commands by running:
`python -m setup.py commandName`.

#### Experiments
Run `python -m setup.py run_kill_experiment` to start a set of experiments.
Options are:
- `-k [y]/n` to use symbolic knowledge injection;
- `-e [y]/n` to use early stop condition.

The command will start the training of 30 neural networks and their test.
Results will be stored in `resources/results/` folder inside `knowledge` or `classic` subfolders.

### Statistics
Run `python -m setup.py get_statistics` to print the statistics for a given experiment population.
Option are:
- `-f [knowledge]/classic` to specify the experiment class.

The command will print several performance metrics values and generate a plot concerning the class accuracy distributions.
The plot will be available in `resources/results` folder.
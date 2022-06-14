# KILL-experiments (WOA 2022)
KILL - Knowledge Injection via Lambda Layer experiments (WOA 2022).

## Users

Users can replicate the results of KILL algorithm applied to the 
primate splice-junction gene sequences dataset available at
https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences.

### Requirements

- python 3.9+
- java 11 (for Antlr4 support)
- antlr4-python3-runtime 4.9.3
- tensorflow 2.7.0
- scikit-learn 1.0.2
- pandas 1.4.2
- numpy 1.22.3
- 2ppy 0.4.0
- psyki 0.1.13
- scipy 1.8.0

### Setup

You can execute a list of predefined commands by running:
`python -m setup.py commandName`.

#### Experiments
Write `python -m setup.py run_kins_experiment` to launch a set of experiments using KINS on the splice junction dataset.
Default options are:
- `-m=fold`, 10-fold cross validation;
- `s=0`, random seed;
- `f=result`, file name to save the results.
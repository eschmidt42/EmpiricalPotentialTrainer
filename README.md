![PottyTrainer](images/logo_2.png)

# PottyTrainer
[![Build Status](https://travis-ci.org/Hamstard/PottyTrainer.svg?branch=master)](https://travis-ci.org/Hamstard/PottyTrainer)

How to train your empirical potential and fit ground state electron density fields.

## Tutorials

### Fitting Electron Densities

Using a cluster-expansion like approach we use 2- and 3-body features to fit electron densities from density functional theory calculations. It can be shown that computing electron densities via contributions of individual neighbouring atoms or groups of neighbouring atoms can be written as a linear regression problem. The pipeline is documented in the `electron density regression tutorial.ipynb` notebook.

### Fitting Energies and Forces using Semi-Empirical Potentials

The functions obtained from fitting electron densities can be used as embedding density functions to develop semi-empirical potentials, e.g. the embedded atom method (EAM). The `EAM regression tutorial.ipynb` documents the individual steps.

## How to cite

TBA

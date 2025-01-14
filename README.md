# Python code for Bayesian fitting of neural tuning curves using all types of regression models. 

## Overview
This package is intended to ease the fitting of Generalized Additive Models.
The code and optimization procedures are all written in Pyro and NumPryo, allowing efficient usage of Probabilitic programming language techniques. 

## Current Models

Supports Poisson and Gaussian with natural hyperparameter tuning via two approaches: sthocastic variational inference or MCMC.
Can you use any type of supplied tensor or regular basis function. 
Naturally implements wiggliness and null space coefficient constraints (a la L1).
Naturally implements (laplacian) gaussian markov field regularization in both 1D and 2D. So, 2D auto regularizaiton for things like place or grid fields is testablre.

## Model Nomenclature


Please direct questions or bugs to justfineneuro@gmail.com, or submit an issue in the GitHub repo!

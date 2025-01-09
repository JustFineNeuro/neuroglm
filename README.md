Python code for doing tuning curves using all types of regression models. 
Supports Poisson and Gaussian with natural hyperparameter tuning via two approaches: sthocastic variational inference or MCMC.
Can you use any type of supplied tensor or regular basis function. 
Naturally implements wiggliness and null space coefficient constraints (a la L1).
Naturally implements (laplacian) gaussian markov field regularization in both 1D and 2D. So, 2D auto regularizaiton for things like place or grid fields is testablre.


TODO: implement GMF across spike history and coupling filters.

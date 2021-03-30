# Project Description
This repository aims to create a model for default payment prediction from the dataset [https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients]()

## Environment setup
In the following lines of commands, you will create the env, install the main packages, and made the env a visible kernel in Jupyter notebooks.
```terminal
$ conda create -n credit_card python=3.6
$ conda activate credit_card
$ pip install pandas
$ conda install -c conda-forge fdasrsf
$ pip install scikit-fda
$ pip install statsmodels
$ pip install ipykernel
$ python -m ipykernel install --user --name=credit_card
$ pip install xlrd
$ pip install imblearn
```

# How to read this project
1. Start by `default_payment.ipynb`, where you can find a full exploratory data analysis, variable selection and two models training with hand-made hyperparameter tuning. At the end, there is an explanition about the model bussines perspective.
1. `gridsearch.py`, where there is a random search for hyperparameters(random sample of 40). The best model is saved in a pickle file.
1. `PCA_with_FDA.ipynb`: FDA works for data exploration of time sequences. And also has a method for dimensionality reduction for time series variables like PY_AMT, and BILL_AMT. A new model can be trained with those variables.

The best model had the following performance for sorting smoothing:

y = -0.0626 + 0.8301x 

r_2=0.8173
"""
This a random gridsearch script that uses sklearn. The kfold cross validation random search
calculate the average of f1 over the n-folds for each combination of hyperparameters. Finally, the best model is saved in "model.pickle"
"""
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
# Calculate f1 score on test data after gridsearch
# create script: function to create train test split, function for griserach, print metrics
# and save best model.

def split_data(file_name, test_size=0.25):
    """This method reads the xls fil, and do the train test split.
    Args:
        file_name (str): excel file name
        test_size (float): traction for test set
    Returns:
        X_train, X_test, y_train, y_test (np.arrays)
    """
    df = pd.read_excel(file_name, header=1)
    # put ID as DataFrame INDEX
    df.set_index('ID', inplace=True)
    drop = ['PAY_AMT5', 'BILL_AMT5','BILL_AMT4','PAY_3','PAY_4',
     'EDUCATION','PAY_6','SEX','MARRIAGE','PAY_5']
    # train test split
    name = 'default payment next month'
    X, y = df.drop([name]+drop, axis=1).values, df[name].values
    return train_test_split(X, y, test_size=test_size, random_state=42)

if __name__ == "__main__":
    
    n_iter = 6
    file_name = "default of credit card clients.xls"
    X_train, X_test, y_train, y_test = split_data(file_name)

    # define hyperparameter space for random search
    dist = dict(n_estimators=[200, 300, 500], max_depth=[7, 13, 15], min_samples_leaf=[5, 10, 20],
               max_samples=[0.6, 0.8], min_samples_split=[10, 6, 20], max_features=[0.7, 0.8])
    print("On process........")
    rfc = RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample', random_state=42)
    clf = RandomizedSearchCV(rfc, dist,refit=True,
                             random_state=42, n_iter=n_iter, scoring='f1', n_jobs=8)
    search = clf.fit(X_train, y_train)

    print(f"\nBest f1(class 1) from Kfold cross validation: {round(search.best_score_, 3)}\n")
    print(search.best_params_)
    y_pred = search.best_estimator_.predict(X_test)
    print("\nCLASSIFICATON REPORT FOR TEST SET:\n")
    print(classification_report(y_test, y_pred))
    # save best model
    pkl_filename = "model.pickle"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(search.best_estimator_, file)
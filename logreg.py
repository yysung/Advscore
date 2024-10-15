# %%
import itertools
import json
import os
import os.path as osp
import sys
import traceback

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import Lasso, LogisticRegression


def sig_stars(s):
    if s <= 0.001:
        return "****"
    if s <= 0.01:
        return "***"
    if s <= 0.05:
        return "**"
    if s <= 0.1:
        return "*"
    return ""


def log_regression_analysis(
    X_train,
    y_train,
    C=1.0,
    penalty="l1",
    fit_intercept=False,
    random_state=0,
    max_iter=1000,
    class_weight="balanced",
):
    model = LogisticRegression(
        random_state=random_state,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        C=C,
        class_weight=class_weight,
        penalty=penalty,
        solver="liblinear",
    )
    model.fit(X_train, y_train)

    # Calculate matrix of predicted class probabilities.
    # Check resLogit.classes_ to make sure that sklearn ordered your classes as expected
    predProbs = model.predict_proba(X_train)
    # # Design matrix -- add column of 1's at the beginning of your X_train matrix
    # X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_design = X_train.values

    # Initiate matrix of 0's, fill diagonal with each predicted observation's variance
    # V = np.diagflat(np.product(predProbs, axis=1))

    # Use np.prod instead of np.product
    V = np.diagflat(np.prod(predProbs, axis=1))

    # Covariance matrix
    covLogit = np.linalg.inv(X_design.T @ V @ X_design)
    # covLogit = np.linalg.inv(np.dot(np.dot(X_design.T, V), X_design))
    # print("Covariance matrix: ", covLogit)

    # Standard errors
    # print("Standard errors: ", np.sqrt(np.diag(covLogit)))

    # Wald statistic (coefficient / s.e.) ^ 2
    diag = np.maximum(np.diag(covLogit), 0)
    std_err = np.sqrt(diag)
    Coefs = model.coef_[0, :]
    Wald = np.abs(Coefs) / (std_err + 1e-12)
    dof = 1
    print(len(X_train) - len(X_train.columns))
    p_values = 2 * stats.norm.cdf(-abs(Wald))
    model_fit = model.score(X_train, y_train) * 100
    _features_list = [*zip(X_train.columns, Coefs, std_err, Wald, p_values)]
    features = []
    for entry in sorted(_features_list, key=lambda x: (-abs(x[1]), x[-1])):
        features.append(entry)
    df = pd.DataFrame(features, columns=["name", "coef", "se", "wald", "pvalue"])
    df.loc[:, "sig"] = df["pvalue"].apply(sig_stars)
    return model_fit, df.set_index("name")


def linear_regression_with_significance(
    df_inputs, labels, alpha=0.1, fit_intercept=False
):
    """
    Perform linear regression with statistical significance analysis.

    Parameters:
    df (pandas.DataFrame): The input dataframe
    feature_columns (list): List of column names to use as features
    label_column (str): Name of the column to use as the label
    alpha (float): Regularization strength for Lasso regression

    Returns:
    tuple: (model_fit, pandas.DataFrame) with model fit score and regression results
    """
    # Prepare the feature matrix X and target vector y
    X = df_inputs
    y = labels

    # Initialize and fit the Lasso model
    model = Lasso(alpha=alpha, random_state=0, fit_intercept=fit_intercept)
    model.fit(X, y)

    # Calculate model fit
    model_fit = model.score(X, y)

    # Extract coefficients
    coefficients = model.coef_

    # Calculate standard errors
    n = len(y)
    p = X.shape[1]
    residuals = y - model.predict(X)
    mse = np.sum(residuals**2) / (n - p - 1)
    X_pseudo_inv = np.linalg.pinv(X.T @ X)
    std_err = np.sqrt(mse * np.diag(X_pseudo_inv))

    # Calculate t-statistic and p-values
    t_statistic = coefficients / std_err
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistic), n - p - 1))

    # Create a list of features with their statistics
    _features_list = [*zip(X.columns, coefficients, std_err, t_statistic, p_values)]
    features = sorted(_features_list, key=lambda x: (-abs(x[1]), x[-1]))

    # Create a dataframe with results
    df = pd.DataFrame(features, columns=["name", "coef", "se", "t", "pvalue"])
    df.loc[:, "sig"] = df["pvalue"].apply(sig_stars)

    return model_fit, df.set_index("name")

# %%

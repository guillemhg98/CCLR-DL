# utils_LMLR.py
# ------------------------------------------------------
# Utility functions for lagged multiple regression models
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# ------------------------------------------------------

import pandas as pd
import numpy as np
import scipy.stats
import math
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# --- Data Preprocessing Utilities ---

def smoother(df, window_size):
    smoothed = {}
    for column in df.columns:
        column_list = []
        for i in range(len(df)):
            if i < window_size:
                column_list.append(df[column].iloc[:i+1].mean())
            else:
                column_list.append(df[column].iloc[i-window_size:i+1].mean())
        smoothed[column] = column_list
    smoothed_df = pd.DataFrame.from_dict(smoothed)
    smoothed_df.set_index(df.index, inplace=True)
    return smoothed_df

def infection_index_df(df, prior_days):
    infection_index = {}
    for column in df.columns:
        column_list = [float('nan') if i < prior_days else df[column].iloc[i] / max(df[column].iloc[(i-prior_days):(i-1)].sum(), 1)
                       for i in range(len(df))]
        infection_index[column] = column_list
    infection_index = pd.DataFrame.from_dict(infection_index)
    infection_index.set_index(df.index, inplace=True)
    return infection_index

# --- Visualization Utilities ---

def plot_example(df, title):
    sampled_cols = np.random.choice(df.columns, size=10, replace=False)
    dff = df[sampled_cols].copy()
    dff["date"] = dff.index
    sns.set(rc={'figure.figsize': (20, 8)})
    sns.lineplot(data=dff.melt(id_vars=['date']), x='date', y='value', hue='variable').set(title=title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def ploter(df, title, n, code):
    cols = list(df.columns)[:n] + [code]
    dff = df[cols]
    dff["date"] = dff.index
    sns.set(rc={'figure.figsize': (20, 8)})
    sns.lineplot(data=dff.melt(id_vars=['date']), x='date', y='value', hue='variable').set(title=title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Correlation and Multicollinearity ---

def get_top_correlations_blog(df, threshold=0.90):
    orig_corr = df.corr()
    abs_corr = orig_corr.abs()
    so = abs_corr.unstack()
    pairs = set()
    result = pd.DataFrame()
    for index, value in so.sort_values(ascending=False).items():
        if value > threshold and index[0] != index[1] and (index[1], index[0]) not in pairs:
            result.loc[len(result)] = [index[0], index[1], orig_corr.loc[index[0], index[1]]]
            pairs.add((index[0], index[1]))
    result.columns = ['Variable 1', 'Variable 2', 'Correlation Coefficient']
    return result.set_index(['Variable 1', 'Variable 2'])

def compute_vif(variables, df):
    X = df[variables].copy()
    X['intercept'] = 1
    vif = pd.DataFrame()
    vif['Variable'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif[vif['Variable'] != 'intercept']

def filter_VIF(vif, df, iterations_max, VIF_threshold):
    redundant_vars = vif.sort_values('VIF', ascending=False)['Variable'].tolist()
    iterations = 0
    while iterations < iterations_max and compute_vif(redundant_vars, df)['VIF'].max() > VIF_threshold:
        redundant_vars.pop(0)
        iterations += 1
    return redundant_vars

# --- Model Training and Evaluation ---

def metrics_calculation(models, y_train, y_test, Y_PRED_train, Y_PRED_test, params):
    results = []
    for i in range(len(models) - 1):
        n = len(y_train)
        k1 = len(models[i].params)
        k2 = len(models[i+1].params)
        rss1, rss2 = models[i].ssr, models[i+1].ssr
        Fstat1 = ((rss1 - rss2)/(k2 - k1)) / (rss2/(n - k2))
        pval1 = 1 - scipy.stats.f.cdf(Fstat1, k2 - k1, n - k2)
        Fstat2 = ((Y_PRED_train[i+1] - y_train.mean()).pow(2).sum() -
                  (Y_PRED_train[i] - y_train.mean()).pow(2).sum()) / (k2 - k1) / (sum((Y_PRED_train[i+1] - y_train)**2) / (n - k2 - 1))
        pval2 = 1 - scipy.stats.f.cdf(Fstat2, k2 - k1, n - k2 - 1)
        results.append({
            'number_of_variables': i,
            'F1': Fstat1,
            'pval1': pval1,
            'F2': Fstat2,
            'pval2': pval2,
            'MAPE_train': mean_absolute_error(y_train, Y_PRED_train[i]) * 100,
            'MAPE_test': mean_absolute_error(y_test, Y_PRED_test[i]) * 100,
            'RMSE_train': math.sqrt(mean_squared_error(y_train, Y_PRED_train[i])),
            'RMSE_test': math.sqrt(mean_squared_error(y_test, Y_PRED_test[i])),
            'predictors': params[i]
        })
    return pd.DataFrame(results)

def models_training(df, code, corr, max_iters):
    Y_PRED_train, Y_PRED_test, models, predictors = [], [], [], []
    corr = corr[:max_iters]
    split_idx = int(len(df) * 0.8)
    y_train = df[code].iloc[:split_idx]
    y_test = df[code].iloc[split_idx:]
    for i in range(1, len(corr)):
        X_train = df[corr.index[:i]].iloc[:split_idx]
        X_test = df[corr.index[:i]].iloc[split_idx:]
        model = sm.OLS(y_train, X_train).fit()
        models.append(model)
        predictors.append(','.join(X_train.columns))
        Y_PRED_train.append(model.predict(X_train))
        Y_PRED_test.append(model.predict(X_test))
    return models, predictors, Y_PRED_train, Y_PRED_test, y_train, y_test

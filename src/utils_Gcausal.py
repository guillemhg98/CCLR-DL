# utils_Gcausal.py
# ------------------------------------------------------
# Utility functions for time series Granger causality analysis
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# Description:
#   This module contains helper functions for smoothing time series,
#   testing stationarity (KPSS), generating lag plots, selecting optimal VAR lags,
#   and computing Granger causality matrices between multiple variables.
# ------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import kpss, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

# --- Preprocessing ---
def smoother(df, window_size):
    smoothed = {}
    for column in df.columns:
        column_list = [df[column].iloc[max(0, i - window_size):i + 1].mean() for i in range(len(df))]
        smoothed[column] = column_list
    smoothed_df = pd.DataFrame(smoothed, index=df.index)
    return smoothed_df

def stationate(df, cols):
    for col in cols:
        df[col] = df[col] - df[col].shift(1)
    return df.dropna()

# --- Visualization ---
def lag_plots(data_df):
    ncol = data_df.shape[1]
    if ncol == 0:
        return
    fig, axes = plt.subplots(1, ncol, figsize=(5 * ncol, 5))
    if ncol == 1:
        axes = [axes]
    for i, col in enumerate(data_df.columns):
        lag_plot(data_df[col], ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_ylabel('$y_{t+1}$')
        axes[i].set_xlabel('$y_t$')
    plt.tight_layout()
    plt.show()

# --- Stationarity Test ---
def kpss_test(data_df):
    test_stat, p_val, cv_1, cv_2_5, cv_5, cv_10 = [], [], [], [], [], []
    valid_cols = []
    for c in data_df.columns:
        series = data_df[c].dropna()
        if series.nunique() <= 1 or len(series) < 10:
            continue
        try:
            res = kpss(series, regression='ct', nlags='auto')
            test_stat.append(res[0])
            p_val.append(res[1])
            cv_1.append(res[3]['1%'])
            cv_2_5.append(res[3]['2.5%'])
            cv_5.append(res[3]['5%'])
            cv_10.append(res[3]['10%'])
            valid_cols.append(c)
        except ValueError:
            continue
    return pd.DataFrame({
        'Test statistic': test_stat,
        'p-value': p_val,
        'Critical value - 1%': cv_1,
        'Critical value - 2.5%': cv_2_5,
        'Critical value - 5%': cv_5,
        'Critical value - 10%': cv_10
    }, index=valid_cols).T.round(4)

# --- Model Selection and G-Causality ---
def splitter(df):
    idx = int(len(df) * 0.8)
    return df.iloc[:idx], df.iloc[idx:]

def select_p(train_df):
    model = VAR(train_df)
    p_range = np.arange(1, 60)
    metrics = {'AIC': [], 'BIC': [], 'FPE': [], 'HQIC': []}
    for p in p_range:
        res = model.fit(p)
        metrics['AIC'].append(res.aic)
        metrics['BIC'].append(res.bic)
        metrics['FPE'].append(res.fpe)
        metrics['HQIC'].append(res.hqic)
    pd.DataFrame(metrics, index=p_range).plot(subplots=True, marker='o', figsize=(15, 10), layout=(2, 2), sharex=True)
    plt.tight_layout()
    plt.show()

def granger_causation_matrix(data, variables, p=1, test='ssr_chi2test'):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], p, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(p)]
            df.loc[r, c] = np.min(p_values)
    df.columns = [f"{var}_x" for var in variables]
    df.index = [f"{var}_y" for var in variables]
    return df

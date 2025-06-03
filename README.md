# CCLR-DL: Hybrid Feature Selection and Forecasting for High-Dimensional Time Series

This repository contains the full implementation of CCLR-DL (Comprehensive Cross-Correlation and Lagged Linear Regression Deep Learning), a hybrid framework combining statistical models and deep learning for feature selection and healthcare demand forecasting.

    📝 This code supports the experiments and results described in the preprint:
    “CCLR-DL: A Novel Statistics and Deep Learning Hybrid Method for Feature Selection and Forecasting Healthcare Demand”
    DOI: 10.20944/preprints202403.1110.v1

## 🔍 Overview

CCLR-DL is a three-phase pipeline designed to:
- Select meaningful predictors from high-dimensional multivariate time series using lagged regression and Granger causality.
- Forecast future values using state-of-the-art deep learning models (e.g. LSTM, GRU, BiLSTM).
- Enhance interpretability and accuracy of long-term forecasting, especially in clinical domains.

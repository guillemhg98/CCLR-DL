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

## 📁 Repository Structure
```
📦CCLR-DL/
 ┣ 📂src/                    # All source code
 ┃ ┣ 📜utils_LMLR.py           # Lagged MLR phase functions
 ┃ ┣ 📜utils_Gcausal.py        # Granger causality phase functions
 ┃ ┣ 📜utils_DL.py             # DL phase functions
 ┃ ┗ 📜synthetic_data.py       # Generator of synthetic dataset
 ┣ 📂notebooks/              # Example walkthroughs and experiments
 ┃ ┣ 📜LMLR_notebook.ipynb     # Lagged MLR
 ┃ ┣ 📜Gcausal_notebook.ipynb  # Granger causality
 ┃ ┣ 📜DL_notebook.ipynb       # Deep learning forecasting
 ┣ 📂data/                   # data
 ┃ ┣ 📜BEST_features_ts350.xlsx   # best features for example ts 350 (most prevalent)
 ┃ ┣ 📂data.zip/ 
 ┃ ┃ ┗ 📜synthetic_timeseries.csv # synthetic 1000 ts dataset
 ┣ 📜main_LMLR.py            # Main runner script LMLR phase
 ┣ 📜main_Gcausal.py         # Main runner script Gcausal phase
 ┣ 📜main_DL.py              # Main runner script DL phase
 ┣ 📜requirements.txt        # Dependencies
 ┗ 📜README.md               # You are here
```

## 📊 Dataset

Due to ethical and legal restrictions, the original clinical dataset (based on 6.3 million patients over 10 years) cannot be made public.

To promote transparency and reproducibility:
- This repository includes a synthetic dataset of 1,000 time series, generated using random resampling techniques.
- The synthetic data replicates key structural characteristics (e.g., sparsity, temporal granularity) of the real data without disclosing any sensitive information.
- 🔒 Real data is hosted on secure institutional servers and cannot be exported nor used without consent.

## 🚀 Getting Started
### 1. Clone the repository
   
    git clone https://github.com/your-org/cclr-dl.git
    cd cclr-dl

### 2. Create environment
   
    pip install -r requirements.txt

### 3. Run an example with synthetic data
    
    python main.py --config configs/example.yaml

## 🧠 Methodology Highlights

- Phase 1 – Feature Selection (Lagged MLR): Identifies non-collinear predictors that best explain the target using forward stepwise regression. It outputs the file "BEST_features_ts350.xlsx" with the best features for each lagging period. 
- Phase 2 – Granger Causality Test: Ensures that selected predictors statistically G-cause the target, increasing explainability. (predictors must be selected from lagged file "BEST_features_ts350.xlsx").
- Phase 3 – Deep Learning Forecasting: Multiple RNN-based architectures are trained using selected features. The best model is selected based on RMSE, MAE, and MAPE. (predictors must be selected from lagged file "BEST_features_ts350.xlsx").

## 📈 Results Summary
- CCLR-DL outperforms baseline models (univariate, random, SHAP) for long-horizon forecasting (≥ 14 days).
- BiLSTM models showed the best overall performance.
- The feature selection phase enhances both performance and interpretability, especially relevant in the healthcare domain

## 🌍 General Applicability

Though developed for healthcare demand modeling, the method is suitable for any high-dimensional time series problem, such as:
- Financial forecasting
- Industrial sensor analysis
- Environmental monitoring

## 🤝 Acknowledgements
Computational resources were partially provided by AQuAS (Catalonia) and the PADRIS program. See paper for full authorship and institutional affiliations.

## 📜 License
This project is licensed under the Apache 2.0 Liccense. See LICENSE file for details.

## 📬 Contact

For questions, please contact the corresponding author:
Guillem Hernández Guillamet — guillemhg98@gmail.com

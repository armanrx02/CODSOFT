# CODSOFT — Data Science Internship

This repository contains work for the CodSoft Data Science Internship. Each task is implemented as a Jupyter notebook that demonstrates data exploration, preprocessing, model training, evaluation, and basic visualizations. Below are brief READMEs for Tasks 1–5 with permalinks to the notebooks.

---

# CODSOFT — Data Science Internship (Task 1)

## Project: Titanic Survival Prediction

This repository contains work for the CodSoft Data Science Internship — Task 1.  
Goal: build a model that predicts whether a passenger on the Titanic survived using the classic Titanic dataset.

### Files of interest

- Notebook: [Task 1.ipynb](https://github.com/armanrx02/CODSOFT/blob/652afd2e4f4cba683f275c08140248c0fcd1d9f0/Task%201.ipynb) — full EDA, preprocessing, Logistic Regression model, evaluation, and visualizations.  
- Dataset: `Titanic-Dataset.csv` (place in repository root or update notebook path)

### Summary

- Model used: Logistic Regression (max_iter=1000)  
- Key preprocessing:
  - Fill missing `Age` with median
  - Fill missing `Embarked` with mode
  - Drop `Cabin`
  - Encode `Sex` (male=0, female=1)
  - One-hot encode `Embarked` (drop_first=True)
  - Drop `PassengerId`, `Name`, `Ticket`
- Train/test split: `test_size=0.2`, `random_state=42`  
- Example evaluation: Accuracy ≈ 0.81; confusion matrix and classification report included in the notebook

### How to run

1. Clone the repository:
   `git clone https://github.com/armanrx02/CODSOFT.git`

2. (Recommended) Create & activate a Python virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies (example):
   `pip install pandas numpy matplotlib seaborn scikit-learn jupyter`  
   Or:
   `pip install -r requirements.txt` (if added)

4. Ensure `Titanic-Dataset.csv` is in the repository root (or update the notebook path).

5. Open and run the notebook:
   `jupyter notebook "Task 1.ipynb"`

### Suggestions / Next steps

- Feature engineering (extract title from `Name`, use `Cabin` deck information)  
- Try other models (RandomForest, XGBoost)  
- Hyperparameter tuning & cross-validation  
- Build a reproducible training script (`train_titanic.py`) and add `requirements.txt`  
- Add unit tests or a small CLI to run training & evaluation outside the notebook

### Contact

For questions about this task or the notebook, contact: armanrx02

---

# CODSOFT — Data Science Internship (Task 2)

## Project: Movie Rating Prediction with Python

This repository contains work for the CodSoft Data Science Internship — Task 2.  
Goal: build a regression model that predicts a movie's rating based on features such as genre, director, actors, duration, and year using the provided IMDb-style dataset.

### Files of interest

- Notebook: [Task 2.ipynb](https://github.com/armanrx02/CODSOFT/blob/f8e91cdcd0f1fed38fdfbd266fd36d13393d0694/Task%202.ipynb) — data loading, cleaning, preprocessing, Linear Regression model, evaluation, and a sample prediction.  
- Dataset: `IMDb Movies India.csv` (place in repository root or update notebook path)

### Summary of approach

- Problem type: Regression (predict movie rating)  
- Model used in the notebook: Linear Regression  
- Preprocessing highlights:
  - Drop rows without `Rating`
  - Fill missing categorical values with `"Unknown"`
  - Clean `Duration` column (`" min"` removal → numeric)
  - Extract/clean `Year` (4-digit) → numeric
  - Encode categorical columns (LabelEncoder used in notebook)
- Train/test split: `test_size=0.2`, `random_state=42`  
- Example evaluation (from the notebook run):  
  - MSE ≈ 1.7461  
  - R² ≈ 0.0608  
  - Example sample prediction shown in the notebook

### How to run

1. Clone the repository:
   `git clone https://github.com/armanrx02/CODSOFT.git`

2. Create & activate a virtual environment and install dependencies (see Task 1 instructions).

3. Place `IMDb Movies India.csv` in the repository root (or update the notebook path).

4. Open and run the notebook:
   `jupyter notebook "Task 2.ipynb"`

### Suggestions / Next steps

- Try stronger models and ensembles (RandomForest, XGBoost, LightGBM)  
- Better categorical handling (target/frequency encoding for high-cardinality fields)  
- Feature engineering (actor/director aggregated stats, decade/age features)  
- Cross-validation and more robust metrics (RMSE, MAE)  
- Convert notebook into reproducible scripts and add `requirements.txt`

### Contact

For questions about this task or the notebook, contact: armanrx02

---

# CODSOFT — Data Science Internship (Task 3)

## Project: Iris Flower Classification

This repository contains work for the CodSoft Data Science Internship — Task 3.  
Goal: train a model that classifies Iris flowers into species (setosa, versicolor, virginica) using sepal and petal measurements.

### Files of interest

- Notebook: [Task 3 - Iris Classification.ipynb](https://github.com/armanrx02/CODSOFT/blob/main/Task%203%20-%20Iris%20Classification.ipynb) — EDA, preprocessing, model training, evaluation, and visualizations.  
- Dataset: `iris.csv` (optional) or use `sklearn.datasets.load_iris`

### Summary

- Dataset: Iris (sepal length/width, petal length/width; species label)  
- Preprocessing:
  - Load dataset (pandas or `sklearn.datasets`)
  - Check for missing values (typically none)
  - Encode target labels (`LabelEncoder` or mapping)
  - Optionally standardize features (`StandardScaler`)
- Models to try:
  - Logistic Regression, k-NN, Decision Tree, Random Forest, SVM
- Train/test split: `test_size=0.2`, `random_state=42`  
- Evaluation metrics: Accuracy, precision, recall, F1-score, confusion matrix  
- Expected performance: often high (commonly > 0.90 with simple models)

### How to run

1. Clone the repository:
   `git clone https://github.com/armanrx02/CODSOFT.git`

2. Create & activate a virtual environment and install dependencies (see Task 1 instructions).

3. If using `iris.csv`, place it in the repository root; otherwise the notebook uses `sklearn`'s built-in dataset.

4. Open and run the notebook:
   `jupyter notebook "Task 3 - Iris Classification.ipynb"`

### Suggestions / Next steps

- Compare several algorithms and cross-validation scores  
- Visualize separability with PCA (2D)  
- Save the best model with `joblib` and add an inference example (`predict_iris.py`)  
- Add unit tests for the preprocessing pipeline

### Contact

For questions about this task or the notebook, contact: armanrx02

---

# CODSOFT — Data Science Internship (Task 4)

## Project: Sales Prediction using Python

This repository contains work for the CodSoft Data Science Internship — Task 4.  
Goal: forecast product sales using historical data and features such as advertising spend, seasonality, promotions, and channels.

### Files of interest

- Notebook: [Task 4 - Sales Prediction.ipynb](https://github.com/armanrx02/CODSOFT/blob/main/Task%204%20-%20Sales%20Prediction.ipynb) — EDA, feature engineering, modeling, and evaluation.  
- Dataset: `sales_data.csv` (optional — place in repository root or update notebook path)

### Summary

- Preprocessing:
  - Handle missing values and outliers
  - Convert date columns to features (month, day-of-week, seasonality)
  - Encode categorical variables (one-hot, target encoding)
  - Scale numeric features if required
- Feature engineering:
  - Lag features (past sales), rolling averages, promotional flags
  - Interaction terms (e.g., spend × channel)
- Models to try:
  - Linear Regression, Ridge/Lasso, RandomForest, XGBoost, LightGBM
  - Time-series approaches when data is sequential (Prophet, ARIMA)
- Evaluation:
  - MAE, RMSE; use time-aware train/validation splits (backtesting) when applicable

### How to run

1. Clone the repository:
   `git clone https://github.com/armanrx02/CODSOFT.git`

2. Create & activate a virtual environment and install dependencies (see Task 1 instructions).

3. Place `sales_data.csv` in the repository root or update the notebook path.

4. Open and run the notebook:
   `jupyter notebook "Task 4 - Sales Prediction.ipynb"`

### Suggestions / Next steps

- Perform extensive feature engineering (lags, rolling stats)  
- Use time-series cross-validation for robust model selection  
- Hyperparameter tuning with `GridSearchCV` / `RandomizedSearchCV`  
- Expose a model via an API (Flask/FastAPI) or build a dashboard for forecasts

### Contact

For questions about this task or the notebook, contact: armanrx02

---

# CODSOFT — Data Science Internship (Task 5)

## Project: Credit Card Fraud Detection

This repository contains work for the CodSoft Data Science Internship — Task 5.  
Goal: build a machine learning model to detect fraudulent credit card transactions.

### Files of interest

- Notebook: [Task 5 - Credit Card Fraud Detection.ipynb](https://github.com/armanrx02/CODSOFT/blob/main/Task%205%20-%20Credit%20Card%20Fraud%20Detection.ipynb) — EDA, preprocessing, modeling, and evaluation.  
- Dataset: `creditcard.csv` (place in repository root or update notebook path)

### Summary

- Challenges:
  - Strong class imbalance (fraud << genuine)
  - Subtle patterns separating fraud from non-fraud
- Preprocessing:
  - Normalize/scale features (`StandardScaler`)
  - Handle class imbalance (undersampling, oversampling, SMOTE)
  - Stratified train/test split
- Models to try:
  - Logistic Regression, RandomForest, XGBoost, LightGBM
  - Anomaly detection approaches (IsolationForest, OneClassSVM) for unsupervised settings
- Evaluation:
  - Precision, recall, F1-score, ROC-AUC (but prioritize precision/recall due to class imbalance)
  - Precision–recall curve and confusion matrix
- Techniques to improve:
  - Resampling (SMOTE/ADASYN), class weighting, threshold tuning, feature selection

### How to run

1. Clone the repository:
   `git clone https://github.com/armanrx02/CODSOFT.git`

2. Create & activate a virtual environment and install dependencies (see Task 1 instructions).

3. Place `creditcard.csv` in the repository root or update the notebook path.

4. Open and run the notebook:
   `jupyter notebook "Task 5 - Credit Card Fraud Detection.ipynb"`

### Suggestions / Next steps

- Try ensemble and tree-based models with class weights  
- Evaluate using cross-validation and stratified folds  
- Create a reproducible training pipeline and save the final model (joblib/pickle)  
- Add an evaluation script that computes precision/recall at multiple thresholds

### Contact

For questions about this task or the notebook, contact: armanrx02

---

## General recommendations (across tasks)

- Add a `requirements.txt` to pin dependency versions.  
- Convert notebooks into Python scripts for reproducible runs (e.g., `train_titanic.py`, `train_iris.py`).  
- Save trained models (joblib/pickle) and include inference examples.  
- Add basic unit tests for preprocessing steps and simple CI checks.  
- Organize datasets under a `data/` folder and update notebook paths accordingly.  
- Add a `LICENSE` (e.g., MIT) if you wish to open-source the work.

Contact
- For questions about these tasks or notebooks, contact: armanrx02

---

End of README

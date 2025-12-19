# CODSOFT — Data Science Internship

Welcome — this repository contains the deliverables for the CodSoft Data Science Internship tasks (Task 1 — Task 5). Each task is provided as a Jupyter notebook that walks through the data exploration, preprocessing, modeling, evaluation and basic visualization for a self-contained ML problem. This README provides an expanded, practical guide so you can reproduce results, extend experiments, and deploy / test models.

Table of contents
- Overview
- Repository structure (files & permalinks)
- Datasets (expected filenames and placement)
- Environments & dependencies (recommended)
- How to run (step-by-step)
- Per-task details (approach, preprocessing, model, hyperparameters, results)
  - Task 1 — Titanic Survival Prediction
  - Task 2 — Movie Rating Prediction
  - Task 3 — Iris Flower Classification
  - Task 4 — Sales Prediction
  - Task 5 — Credit Card Fraud Detection
- Reproducibility & productionization suggestions
- How to contribute
- License & contact

---

Overview
This repo demonstrates end-to-end workflows for common supervised learning tasks:
- binary classification (Titanic, Fraud)
- multiclass classification (Iris)
- regression (Movie rating, Sales)

Each notebook includes:
- EDA and sanity-checks
- Data cleaning & feature preprocessing
- Train/test split and model training
- Evaluation with appropriate metrics
- Visualizations or example predictions

Repository structure (main files)
- Notebooks (per-task, permalinks):
  - Task 1 — Titanic Survival Prediction: codsoft_task_1.ipynb  
    https://github.com/armanrx02/CODSOFT/blob/fa04fe94fcb5e9283024752f59d55947f8ba05e9/codsoft_task_1.ipynb
  - Task 2 — Movie Rating Prediction: codsoft_task_2.ipynb  
    https://github.com/armanrx02/CODSOFT/blob/fa04fe94fcb5e9283024752f59d55947f8ba05e9/codsoft_task_2.ipynb
  - Task 3 — Iris Flower Classification: codsoft_task_3.ipynb  
    https://github.com/armanrx02/CODSOFT/blob/fa04fe94fcb5e9283024752f59d55947f8ba05e9/codsoft_task_3.ipynb
  - Task 4 — Sales Prediction: codsoft_task_4.ipynb  
    https://github.com/armanrx02/CODSOFT/blob/fa04fe94fcb5e9283024752f59d55947f8ba05e9/codsoft_task_4.ipynb
  - Task 5 — Credit Card Fraud Detection: codsoft_task_5.ipynb  
    https://github.com/armanrx02/CODSOFT/blob/fa04fe94fcb5e9283024752f59d55947f8ba05e9/codsoft_task_5.ipynb

- Data files (expected names; place these in repository root or update notebook paths):
  - Titanic-Dataset.csv
  - IMDb Movies India.csv
  - IRIS.csv (or use sklearn dataset inside notebook)
  - advertising.csv (or sales_data.csv)
  - creditcard.csv

Datasets
- Keep datasets under a data/ directory for clarity (recommended). If you keep them at repo root, the notebooks should still run as-is (paths in notebooks assume root).
- If a dataset is large, add instructions to download from the original source and include a small sample for quick runs.

Environment & dependencies
- Recommended workflow (Unix/macOS shown, Windows equivalent provided where needed):
  1. Create virtual environment:
     - python -m venv venv
     - source venv/bin/activate  # macOS / Linux
     - venv\Scripts\activate     # Windows
  2. Install dependencies:
     - pip install -r requirements.txt
- Example requirements (create a file requirements.txt in repo root):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - jupyter
  - imbalanced-learn
  - joblib
  - (optional) xgboost, lightgbm
- To capture exact environment after testing:
  - pip freeze > requirements.txt

How to run (quick)
1. Clone the repo:
   - git clone https://github.com/armanrx02/CODSOFT.git
2. Create & activate venv, install dependencies (see above).
3. Ensure required dataset files are in the expected paths (notebooks expect the CSV filenames listed above).
4. Launch notebooks:
   - jupyter notebook
   - Open the notebook you want to run (e.g., codsoft_task_1.ipynb) and run cells top-to-bottom.

Per-task details (expanded)

Task 1 — Titanic Survival Prediction
- Notebook: codsoft_task_1.ipynb (permalink above)
- Problem: binary classification (Survived: 0/1)
- Key preprocessing:
  - Fill missing Age with median
  - Fill missing Embarked with mode
  - Drop Cabin (too many missing)
  - Map Sex: male=0, female=1
  - One-hot encode Embarked (drop_first=True)
  - Drop PassengerId, Name, Ticket (not used)
- Features used: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_* (one-hot)
- Model: LogisticRegression(max_iter=1000)
- Train/test split: test_size=0.2, random_state=42
- Example results from the notebook:
  - Accuracy ≈ 0.8101
  - Confusion matrix & classification report included
- Suggested improvements:
  - Extract title from Name (Mr/Mrs/Miss/Master → useful feature)
  - Derive Cabin deck (A/G) from Cabin and fill blanks by heuristics
  - Use tree-based models (RandomForest, XGBoost) and tune hyperparameters
  - Perform cross-validation and nested CV for robust model selection
  - Save final model: joblib.dump(model, "titanic_model.joblib")
  - Create train_titanic.py for reproducibility

Task 2 — Movie Rating Prediction (IMDb-style)
- Notebook: codsoft_task_2.ipynb
- Problem: regression (predict Rating)
- Data cleaning highlights:
  - Drop rows with missing Rating (target)
  - Clean Duration (remove "min"), convert to numeric, fill missing with median
  - Extract 4-digit Year and convert to numeric
  - Fill missing categorical values with "Unknown"
  - LabelEncoder used for Genre, Director, Actor 1 (not ideal for production; consider one-hot or target encoding)
- Model: LinearRegression (baseline)
- Train/test split: test_size=0.2, random_state=42
- Example results:
  - MSE ≈ 1.7461, R² ≈ 0.0608 (shows linear model captures small portion of variance)
- Suggested improvements:
  - Use ensemble regressors (RandomForest, XGBoost, LightGBM)
  - Use target encoding / frequency encoding for high-cardinality categorical columns
  - Extract more features from text fields (genre tokens, actor/director popularity)
  - Use cross-validation and more robust metrics (RMSE, MAE)
  - Consider stacking or ensembling multiple models

Task 3 — Iris Flower Classification
- Notebook: codsoft_task_3.ipynb
- Problem: multiclass classification (setosa, versicolor, virginica)
- Clean/prepare:
  - Dataset has 4 numeric features (sepal/petal lengths & widths)
  - Typical preprocessing: (optional) StandardScaler
- Model: LogisticRegression (max_iter tuned in notebook)
- Train/test split: test_size=0.2, random_state=42
- Example results:
  - Accuracy reported 1.0 on test set in the notebook (likely due to small dataset and chosen split; ensure cross-validation to check generalization)
- Suggestions:
  - Try k-NN, DecisionTree, RandomForest, SVM and compare with cross-validation
  - Use PCA for 2D visualization and dataset understanding
  - Save a trained model and add an inference script predict_iris.py

Task 4 — Sales Prediction (Advertising dataset)
- Notebook: codsoft_task_4.ipynb
- Problem: regression to predict Sales from advertising spend (TV, Radio, Newspaper)
- Preprocessing:
  - Use raw numeric features; check for outliers
  - Consider scaling if model needs it (not strictly necessary for tree models)
- Model: LinearRegression (baseline)
- Train/test split: test_size=0.2, random_state=42
- Example results:
  - MSE ≈ 2.9078, R² ≈ 0.9059 (strong fit)
- Suggestions:
  - Try regularized linear models, ensembles (RandomForest/XGBoost) for non-linear relationships
  - If data is time-series-based, use time-aware splits (backtesting)
  - Build a small API endpoint to serve predictions (Flask/FastAPI)

Task 5 — Credit Card Fraud Detection
- Notebook: codsoft_task_5.ipynb
- Problem: highly imbalanced binary classification (Class: 0 = genuine, 1 = fraud)
- Preprocessing:
  - Scale Amount with StandardScaler
  - Use SMOTE to oversample the minority class on the training set
  - Stratified train/test split
- Model: LogisticRegression (solver='liblinear', class_weight='balanced') after SMOTE
- Example results:
  - The notebook shows very high overall accuracy (driven by the majority). Focus on minority-class metrics:
    - Precision for fraud low; recall high in the example (depends on threshold)
    - Examine PR curve and set thresholds to tune precision/recall tradeoff
- Suggested improvements:
  - Use tree-based models with class weights, or tuned XGBoost/LightGBM
  - Use Time or sequence features for user-level aggregation if available
  - Try anomaly-detection algorithms (IsolationForest) if labels are limited
  - Evaluate using precision-recall AUC and confusion matrices across thresholds

Reproducibility & productionization suggestions
- Add a `requirements.txt` and (optionally) `environment.yml` for conda.
- Add scripts:
  - train_<task>.py to run preprocessing, training and save models
  - evaluate_<task>.py to compute metrics on a holdout set and save artifacts (confusion matrix, classification report, plots)
  - predict_<task>.py or a minimal REST API (FastAPI) to serve predictions
- Model saving:
  - Use joblib.dump(model, "models/<model_name>.joblib")
  - Save preprocessing pipeline (ColumnTransformer / Pipeline) together with model
- CI & tests:
  - Add small unit tests for preprocessing steps (test that missing values are filled and shapes are correct)
  - Add a GitHub Actions workflow to run linting and simple tests on push
- Experiment tracking:
  - Use MLflow or a simple CSV logger to track hyperparameters & metrics across runs

How to contribute
- Fork the repository and open a pull request
- Suggested contributions:
  - Add `requirements.txt` and pin versions
  - Convert notebook experiments to Python scripts (reproducible CLI)
  - Add more robust feature engineering, hyperparameter search, and result logging
  - Add unit tests and a CI workflow
- When submitting PRs, please:
  - Add a short description of changes
  - Include reproducible instructions if you add new notebooks or scripts

License
- No license file is provided in the repo by default. If you want to share this work publicly, include a LICENSE (MIT recommended for open-source demos).

Contact
- For questions about these tasks or notebooks, contact: armanrx02

---

Notes and tips
- When running notebooks, rerun kernel & run all cells top-to-bottom to ensure the outputs are recreated in order.
- If you change dataset locations, update the file paths near the top of each notebook (there are comments indicating where to change file paths).
- For experimentation, keep datasets in data/ and use relative paths (data/Titanic-Dataset.csv) to keep the repo tidy.
- Cross-validate model results and avoid reporting single-split metrics as definitive — prefer k-fold CV or repeated holdout where possible.

End of README

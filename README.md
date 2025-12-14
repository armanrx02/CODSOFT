# CODSOFT — Data Science Internship (Task 2)

## Project: Movie Rating Prediction with Python

This repository contains work for the CodSoft Data Science Internship — Task 2.  
Goal: build a regression model that predicts a movie's rating based on features such as genre, director, actors, duration, and year using the provided IMDb-style dataset.

### Files of interest

- Task 2.ipynb — Jupyter notebook containing data loading, cleaning, preprocessing, model training (Linear Regression), evaluation, and a sample prediction. Notebook permalink (commit): https://github.com/armanrx02/CODSOFT/blob/f8e91cdcd0f1fed38fdfbd266fd36d13393d0694/Task%202.ipynb
- IMDb Movies India.csv — dataset used by the notebook (should be placed in the repository root or the path expected by the notebook).

### Summary of approach

- Problem type: Regression (predict movie rating)
- Model used in the notebook: Linear Regression
- Feature selection (example used in notebook):
  - Genre (encoded)
  - Director (encoded)
  - Actor 1 (encoded)
  - Duration (numeric, minutes)
  - Year (numeric)
- Preprocessing & cleaning highlights:
  - Drop rows without Rating
  - Fill missing categorical values (Genre, Director, Actor 1) with "Unknown"
  - Clean Duration column by removing " min" and converting to numeric; fill missing durations with median
  - Extract 4-digit year from Year column and convert to numeric; fill missing years with median
  - Encode categorical columns with LabelEncoder (note: for production, consider one-hot encoding or target encoding depending on model)
- Train/test split:
  - test_size = 0.2, random_state = 42

### Example evaluation (from the notebook run)

- Mean Squared Error (MSE): ~1.7461  
- R² score: ~0.0608  
- Example sample prediction (duration 120, year 2022, using dataset means for encoded features): Predicted Rating ≈ 5.54

(These numbers indicate the linear model captures only a small portion of rating variance — see suggestions below to improve.)

### How to run

1. Clone the repository:
   git clone https://github.com/armanrx02/CODSOFT.git

2. (Recommended) Create & activate a Python virtual environment:
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate      # Windows

3. Install dependencies (example):
   pip install pandas numpy scikit-learn jupyter

   Or if a requirements.txt is present:
   pip install -r requirements.txt

4. Place the dataset (IMDb Movies India.csv) in the repository root (or update the notebook path).

5. Open and run the notebook:
   jupyter notebook "Task 2.ipynb"

### Suggestions / Next steps

- Try stronger models and ensembles:
  - RandomForestRegressor, GradientBoosting / XGBoost / LightGBM
  - Regularized linear models (Ridge, Lasso)
- Better categorical handling:
  - Use target encoding or frequency encoding for high-cardinality categorical fields (Director, Actors)
  - Consider extracting features from textual fields (e.g., split Genre into multiple binary genre columns)
- Feature engineering:
  - Extract title tokens, popularity features (if Votes available), or compute actor/director average ratings
  - Use release decade or relative year (age of movie) instead of raw year
- Evaluation improvements:
  - Use cross-validation (K-fold) for robust validation
  - Try RMSE and MAE in addition to MSE/R²
- Reproducibility:
  - Add a requirements.txt
  - Convert notebook steps into a script (train.py) that accepts arguments and saves trained models
- Data quality:
  - Clean and normalize Votes, handle outliers in Duration and Rating
  - If more metadata is available, include features such as language, country, or budget

### Contact

For questions about this task or the notebook, contact: armanrx02

---
(End of README)

# CODSOFT — Data Science Internship (Task 1)

## Project: Titanic Survival Prediction

This repository contains work for the CodSoft Data Science Internship — Task 1.  
Goal: build a model that predicts whether a passenger on the Titanic survived using the classic Titanic dataset.

### Files of interest

- Task 1.ipynb — Jupyter notebook with the full data exploration, preprocessing, model training (Logistic Regression), evaluation, and visualizations for the Titanic Survival Prediction task. Notebook (permalink): https://github.com/armanrx02/CODSOFT/blob/69d3edc32be2d3fdfb6da7ebfe7a6050792a59ff/Task%201.ipynb  
- Titanic-Dataset.csv — dataset used by the notebook (should be placed in the repository root or the path expected by the notebook).

### Summary

- Model used: Logistic Regression (max_iter=1000)  
- Key preprocessing:
  - Fill missing Age with median
  - Fill missing Embarked with mode
  - Drop Cabin (many missing values)
  - Encode Sex (male=0, female=1)
  - One-hot encode Embarked (drop_first=True)
  - Drop PassengerId, Name, Ticket
- Train/test split: test_size=0.2, random_state=42
- Example notebook evaluation (from Task 1.ipynb): Accuracy ≈ 0.81; confusion matrix and classification report included in the notebook.

### How to run

1. Clone the repository:
   git clone https://github.com/armanrx02/CODSOFT.git

2. (Recommended) Create & activate a Python virtual environment:
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows

3. Install dependencies (example):
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter

   Or if you add a requirements.txt:
   pip install -r requirements.txt

4. Ensure Titanic-Dataset.csv is in the repository root (or update the notebook path).

5. Open and run the notebook:
   jupyter notebook "Task 1.ipynb"

### Suggestions / Next steps

- Improve model performance with:
  - Feature engineering (extract title from Name, use Cabin deck information)
  - Try other models (RandomForest, XGBoost)
  - Hyperparameter tuning & cross-validation
  - Build a reproducible training script (train.py) and add requirements.txt
- Add unit tests or a small CLI to run training & evaluation outside the notebook.

### Contact

For questions about this task or the notebook, contact: armanrx02

---
(End of README)

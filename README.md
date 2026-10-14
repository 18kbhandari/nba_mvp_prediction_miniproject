# NBA MVP Prediction Using Machine Learning

## Project Overview
This mini-project explores NBA MVP prediction using historical player-season statistics and historical MVP voting records. The notebook compares four classifiers and produces a candidate ranking.

## Models
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)

## Dataset Files
Place these files in the `data/` folder:
- `Player Per Game.csv`
- `Player Award Shares.csv`

The dataset combines player performance statistics with MVP award results. The MVP target is binary: `1` for an MVP winner and `0` otherwise.

## Features
The initial models use age, games played, games started, minutes per game, field-goal percentage, three-point percentage, free-throw percentage, rebounds, assists, steals, blocks, turnovers, and points per game.

## Evaluation
The notebook compares accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices. It also examines feature importance and generates a ranking using the average ranks of the four models.

## Repository Structure
```text
nba_mvp_prediction_miniproject/
├── data/
│   ├── Player Per Game.csv
│   └── Player Award Shares.csv
├── notebooks/
│   └── NBA_MVP_Prediction.ipynb
├── outputs/
│   ├── nba_mvp_model_data.csv
│   ├── final_mvp_ranking.csv
│   └── *_feature_importance.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup and Run
1. Install Python 3.10 or newer.
2. From the repository root, install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Start Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open `notebooks/NBA_MVP_Prediction.ipynb` and run the cells from top to bottom.

Run the notebook with its working directory set to the `notebooks/` folder. It reads source data from `../data/` and writes generated CSV results to `../outputs/`.

## Outputs
- `final_mvp_ranking.csv`: candidate ranking from the four-model average-rank ensemble.
- `nba_mvp_model_data.csv`: cleaned data used for modeling.
- `random_forest_feature_importance.csv`
- `decision_tree_feature_importance.csv`
- `logistic_regression_feature_importance.csv`
- `svm_feature_importance.csv`

## Limitations
MVP winners are rare compared with non-winners, causing significant class imbalance. Reported metrics should be interpreted cautiously, especially because the chronological test set contains only a small number of MVP winners. The ranking is an experimental model output, not a guarantee of the actual award result.

## Technologies
Python, Pandas, NumPy, scikit-learn, Matplotlib, Jupyter Notebook.

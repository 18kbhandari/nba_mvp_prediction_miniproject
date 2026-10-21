# NBA MVP Prediction Using Machine Learning

## Project Overview
This mini-project explores NBA MVP prediction using historical player-season statistics and MVP award records. It prepares the data, trains four classifiers, evaluates their performance on later seasons, and produces a ranking of MVP candidates.

## Models
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)

A majority-class baseline is also included for comparison.

## Dataset Files
Place these files in the `data/` folder:
- `Player Per Game.csv`
- `Player Award Shares.csv`

The player statistics are combined with historical MVP award results. The target is binary: `1` for an MVP winner and `0` for other players in the included seasons. The notebook checks for duplicate player-season records and handles missing feature values without dropping MVP winners.

## Features
The models use age, games played, games started, minutes per game, field-goal percentage, three-point percentage, free-throw percentage, rebounds, assists, steals, blocks, turnovers, and points per game.

## Evaluation
The notebook uses a chronological split, training on seasons through 2021 and testing on seasons from 2022 to 2025. It reports accuracy, precision, recall, F1-score, ROC-AUC, average precision, and confusion matrices. It also evaluates season-level rankings and exports feature-importance results.

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
│   ├── model_evaluation.csv
│   ├── season_ranking_evaluation.csv
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
- `final_mvp_ranking.csv`: ranked MVP candidates based on the model outputs.
- `nba_mvp_model_data.csv`: cleaned dataset used for modeling.
- `model_evaluation.csv`: classification metrics for the models and baseline.
- `season_ranking_evaluation.csv`: ranking performance for the held-out seasons.
- `decision_tree_feature_importance.csv`
- `logistic_regression_feature_importance.csv`
- `random_forest_feature_importance.csv`
- `svm_feature_importance.csv`

## Limitations
MVP winners are rare compared with non-winners, causing substantial class imbalance. 
The held-out test period contains only four seasons, from 2022 to 2025, with one MVP winner per season. 
The combined ranking places the actual winner first in two of those four seasons and within the top five in all four. 
These results are preliminary and should not be treated as proof of reliable future performance. 
The combined score is a ranking score, not a calibrated probability of winning MVP.

## Technologies
Python, Pandas, NumPy, scikit-learn, Matplotlib, Jupyter Notebook.

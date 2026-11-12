# NBA MVP Prediction: Controlled Model Selection

This end-of-season project predicts one NBA MVP winner per season from player statistics. It uses the supplied player and MVP award files without modifying them. The original notebook remains in `notebooks/NBA_MVP_Prediction_original_backup.ipynb`.

## Final experimental protocol

The final revision uses only four models: Logistic Regression, RBF SVM, Random Forest, and XGBoost. It uses shared 2018–2024 walk-forward validation folds and treats 2025 as an untouched chronological holdout.

The experiments change one factor at a time:

1. Training windows: 10, 15, 20, 25, and all available prior seasons.
2. Feature configuration: the original 13 statistics versus those statistics plus seven leakage-safe player-history features.
3. Candidate population: all players, top 10, and top 5 current-season performance candidates.

Candidate pools use only current-season points, assists, and rebounds. They never use award shares, MVP labels, or future information. The output reports MVP inclusion rate so filtering can be audited.

## Results from the supplied data

Validation chose full prior history, the original 13 features, and a top-five candidate pool. Logistic Regression was selected because it had the highest primary validation metric: **71.4% Top-1 accuracy** (five of seven 2018–2024 seasons), with 85.7% Top-3 accuracy and PR-AUC 0.605.

Player-history features did not improve the final selected model. Team configuration C was not run because the repository does not contain reliable team wins, win percentage, or conference-standing data; no values were fabricated.

On the untouched 2025 holdout, Random Forest selected the actual MVP first. This one-season result is reported separately and was not used for method selection.

## Leakage protections

- Keys are `(season, player_id)`; `2TM` rows are preferred for traded players.
- Career and prior-season variables use `shift(1)` within `player_id`, so a current season never contributes to its own history.
- Rookie rows remain, with missing history handled by fold-fitted imputers.
- The 2026 incomplete season is excluded.
- Each walk-forward fold fits preprocessing and its model only on earlier seasons.

## Run

```powershell
python -m pip install -r requirements.txt
python final_revision.py
```

Or execute `notebooks/NBA_MVP_Prediction.ipynb`.

## Outputs

The `outputs/` folder contains training-window, feature, candidate-population, and four-model comparisons; season-wise validation and holdout predictions; final model selection; feature importance; a confusion matrix; and precision-recall curves.

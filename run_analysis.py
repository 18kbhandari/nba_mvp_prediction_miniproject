"""Leakage-safe, end-of-season NBA MVP analysis.

Run from the repository root: python run_analysis.py
The two files in data/ are read only; all derived artefacts go to outputs/.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve)
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42
BASE_FEATURES = ["age", "g", "gs", "mp_per_game", "fg_percent", "x3p_percent", "ft_percent",
                 "trb_per_game", "ast_per_game", "stl_per_game", "blk_per_game", "tov_per_game", "pts_per_game"]
HISTORY_FEATURES = ["seasons_played", "career_games", "career_minutes", "is_rookie", "career_start_season",
 "career_avg_points", "career_avg_assists", "career_avg_rebounds", "career_avg_minutes", "career_avg_steals",
 "career_avg_blocks", "career_avg_fg_percent", "career_avg_3p_percent", "career_avg_ft_percent",
 "previous_season_points", "previous_season_assists", "previous_season_rebounds", "previous_season_minutes",
 "previous_season_games", "previous_season_fg_percent", "previous_season_3p_percent", "previous_season_ft_percent",
 "last_3_seasons_avg_points", "last_3_seasons_avg_assists", "last_3_seasons_avg_rebounds", "last_3_seasons_avg_minutes",
 "points_improvement", "assists_improvement", "rebounds_improvement", "career_points_std", "has_previous_history"]

def safe_num(frame, cols):
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame

def clean_and_engineer():
    """Make one row per player/season, then construct only strictly prior history."""
    p = pd.read_csv(DATA / "Player Per Game.csv", na_values=["NA", ""])
    a = pd.read_csv(DATA / "Player Award Shares.csv", na_values=["NA", ""])
    p = safe_num(p, ["season"] + BASE_FEATURES)
    p = p.dropna(subset=["season", "player_id"]).copy()
    p["season"] = p.season.astype(int)
    # 2TM is the published total; otherwise aggregate rate rows by games played, avoiding duplicate players.
    p["_total"] = p.team.astype(str).eq("2TM")
    p = p.sort_values(["season", "player_id", "_total", "g"], ascending=[True, True, False, False])
    p = p.drop_duplicates(["season", "player_id"], keep="first").drop(columns="_total")
    a["season"] = pd.to_numeric(a["season"], errors="coerce")
    mvp = a[a["award"].astype(str).str.lower().eq("nba mvp")].copy()
    mvp = mvp.dropna(subset=["season", "player_id"])
    mvp["season"] = mvp.season.astype(int)
    mvp["winner"] = mvp.winner.astype(str).str.upper().eq("TRUE").astype(int)
    # A season is eligible only if the supplied awards data confirms exactly one MVP winner.
    confirmed = mvp.groupby("season")["winner"].sum()
    completed = confirmed[confirmed.eq(1)].index
    mvp = mvp[mvp.season.isin(completed)].groupby(["season", "player_id"], as_index=False).agg(
        voting_share=("share", "max"), MVP=("winner", "max"))
    d = p.merge(mvp, on=["season", "player_id"], how="left", validate="one_to_one")
    d["MVP"] = d["MVP"].fillna(0).astype(int)
    # Zero shares are defensible only for these completed MVP-award seasons.
    d["voting_share"] = d["voting_share"].where(d.season.isin(completed))
    d.loc[d.season.isin(completed), "voting_share"] = d.loc[d.season.isin(completed), "voting_share"].fillna(0)
    d = d.sort_values(["player_id", "season"]).reset_index(drop=True)
    hist_cols = {"g":"career_games", "mp_per_game":"career_minutes", "pts_per_game":"career_avg_points",
      "ast_per_game":"career_avg_assists", "trb_per_game":"career_avg_rebounds", "stl_per_game":"career_avg_steals",
      "blk_per_game":"career_avg_blocks", "fg_percent":"career_avg_fg_percent", "x3p_percent":"career_avg_3p_percent",
      "ft_percent":"career_avg_ft_percent"}
    d["career_start_season"] = d.groupby("player_id").season.transform("min")
    d["seasons_played"] = d.groupby("player_id").cumcount()
    d["is_rookie"] = d.seasons_played.eq(0).astype(int)
    d["has_previous_history"] = d.is_rookie.eq(0).astype(int)
    # shift before expanding/rolling: current row is never included.
    for source, target in hist_cols.items():
        prior = d.groupby("player_id")[source].shift(1)
        if source == "g": d[target] = prior.groupby(d.player_id).cumsum().fillna(0)
        elif source == "mp_per_game":
            # accumulated minutes proxy: prior season games x minutes/game
            season_minutes = d["g"] * d["mp_per_game"]
            d[target] = season_minutes.groupby(d.player_id).shift(1).groupby(d.player_id).cumsum().fillna(0)
            d["career_avg_minutes"] = prior.groupby(d.player_id).expanding().mean().reset_index(level=0, drop=True)
        else: d[target] = prior.groupby(d.player_id).expanding().mean().reset_index(level=0, drop=True)
    previous = {"pts_per_game":"previous_season_points", "ast_per_game":"previous_season_assists",
      "trb_per_game":"previous_season_rebounds", "mp_per_game":"previous_season_minutes", "g":"previous_season_games",
      "fg_percent":"previous_season_fg_percent", "x3p_percent":"previous_season_3p_percent", "ft_percent":"previous_season_ft_percent"}
    for source, target in previous.items(): d[target] = d.groupby("player_id")[source].shift(1)
    for source, target in [("pts_per_game", "last_3_seasons_avg_points"), ("ast_per_game", "last_3_seasons_avg_assists"),
                           ("trb_per_game", "last_3_seasons_avg_rebounds"), ("mp_per_game", "last_3_seasons_avg_minutes")]:
        d[target] = d.groupby("player_id")[source].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    for cur, prior, target in [("pts_per_game","previous_season_points","points_improvement"),
                               ("ast_per_game","previous_season_assists","assists_improvement"),
                               ("trb_per_game","previous_season_rebounds","rebounds_improvement")]: d[target] = d[cur] - d[prior]
    d["career_points_std"] = d.groupby("player_id")["pts_per_game"].transform(lambda s: s.shift(1).expanding().std())
    assert not d.duplicated(["season", "player_id"]).any()
    return d, sorted(completed)

def make_models():
    return {
      "Logistic Regression": Pipeline([("imputer",SimpleImputer(strategy="median", add_indicator=True)),("scaler",StandardScaler()),("model",LogisticRegression(class_weight="balanced",max_iter=1000,random_state=SEED))]),
      "Gaussian Naive Bayes": Pipeline([("imputer",SimpleImputer(strategy="median", add_indicator=True)),("scaler",StandardScaler()),("model",GaussianNB())]),
      "Decision Tree": Pipeline([("imputer",SimpleImputer(strategy="median", add_indicator=True)),("model",DecisionTreeClassifier(class_weight="balanced",max_depth=5,min_samples_leaf=8,random_state=SEED))]),
      "Random Forest": Pipeline([("imputer",SimpleImputer(strategy="median", add_indicator=True)),("model",RandomForestClassifier(n_estimators=300,class_weight="balanced",max_depth=10,min_samples_split=8,min_samples_leaf=3,random_state=SEED,n_jobs=-1))]),
      "SVM": Pipeline([("imputer",SimpleImputer(strategy="median", add_indicator=True)),("scaler",StandardScaler()),("model",SVC(kernel="rbf",class_weight="balanced",probability=True,random_state=SEED))])}

def walk_forward(data, features, completed):
    eligible = data[data.season.isin(completed)].copy()
    seasons = sorted(eligible.season.unique())
    # Start once there are at least eight labelled seasons and a positive class in training.
    # Modern columns are not consistently recorded in the oldest supplied seasons.
    # 2018 onward gives the requested expanding-window evaluation and complete modern inputs.
    tests = [s for s in seasons if s >= 2018 and sum(x < s for x in seasons) >= 8]
    collected=[]
    for test_year in tests:
        train, test = eligible[eligible.season < test_year], eligible[eligible.season.eq(test_year)]
        for name, model in make_models().items():
            model.fit(train[features], train.MVP)
            prob = model.predict_proba(test[features])[:,1]
            out=test[["season","player_id","player","team","MVP","voting_share"]].copy(); out["model"]=name; out["probability"]=prob
            collected.append(out)
    return pd.concat(collected, ignore_index=True), tests

def metric_rows(preds, experiment):
    rows=[]
    for name, x in preds.groupby("model"):
        y, score=x.MVP, x.probability
        # Threshold is selected from OOF predictions; this table reports all candidate thresholds.
        for threshold in [.30,.40,.50,.60,.70,.80]:
            z=(score>=threshold).astype(int); cm=confusion_matrix(y,z,labels=[0,1])
            rows.append(dict(Model=name, Experiment=experiment, Threshold=threshold, Accuracy=accuracy_score(y,z), Precision=precision_score(y,z,zero_division=0), Recall=recall_score(y,z,zero_division=0), F1=f1_score(y,z,zero_division=0), **{"ROC-AUC":roc_auc_score(y,score),"PR-AUC":average_precision_score(y,score),"False positives":int(cm[0,1]),"True positives":int(cm[1,1])}))
    return pd.DataFrame(rows)

def ranking_metrics(preds, experiment):
    ranked=preds.copy(); ranked["predicted_rank"]=ranked.groupby(["model","season"]).probability.rank(method="min",ascending=False).astype(int)
    winners=ranked[ranked.MVP.eq(1)]
    by=winners.groupby("model").agg(**{"Top-1 accuracy":("predicted_rank",lambda s:(s<=1).mean()),"Top-3 accuracy":("predicted_rank",lambda s:(s<=3).mean()),"Top-5 accuracy":("predicted_rank",lambda s:(s<=5).mean()),"MRR":("predicted_rank",lambda s:(1/s).mean()),"Seasons":("season","nunique")}).reset_index()
    by["Experiment"]=experiment
    return ranked,by

def plot_curves(preds, label):
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for name,x in preds.groupby("model"):
        pr,re,_=precision_recall_curve(x.MVP,x.probability); ax[0].plot(re,pr,label=name)
        fpr,tpr,_=roc_curve(x.MVP,x.probability); ax[1].plot(fpr,tpr,label=name)
    ax[0].set(xlabel="Recall",ylabel="Precision",title=f"{label}: precision-recall"); ax[1].set(xlabel="False positive rate",ylabel="True positive rate",title=f"{label}: ROC")
    for a in ax: a.legend(fontsize=8); a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT/f"{label.lower().replace(' ','_')}_curves.png",dpi=160); plt.close(fig)

def feature_importance(data, features, completed):
    latest=max(completed); train=data[(data.season.isin(completed))&(data.season<latest)]
    rows=[]
    models=make_models()
    for name in models:
        models[name].fit(train[features],train.MVP)
    for name in ["Random Forest","Decision Tree"]:
        imp=models[name].named_steps["model"].feature_importances_[:len(features)]
        rows.extend({"Model":name,"Feature":f,"Importance":v} for f,v in zip(features,imp))
    coef=models["Logistic Regression"].named_steps["model"].coef_[0][:len(features)]
    rows.extend({"Model":"Logistic Regression","Feature":f,"Importance":v} for f,v in zip(features,coef))
    sample=data[(data.season.eq(latest))&(data.season.isin(completed))]
    pi=permutation_importance(models["SVM"],sample[features],sample.MVP,n_repeats=10,random_state=SEED,scoring="average_precision")
    rows.extend({"Model":"SVM (permutation)","Feature":f,"Importance":v} for f,v in zip(features,pi.importances_mean))
    out=pd.DataFrame(rows); out.to_csv(OUT/"feature_importance.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,7)); top=out[out.Model.eq("Random Forest")].nlargest(15,"Importance").sort_values("Importance")
    ax.barh(top.Feature,top.Importance); ax.set_title("Random Forest feature importance (improved features)"); fig.tight_layout(); fig.savefig(OUT/"feature_importance_random_forest.png",dpi=160); plt.close(fig)

def main():
    data, completed = clean_and_engineer()
    # 2026 player data is intentionally excluded because it has no confirmed MVP award.
    data.to_csv(OUT/"cleaned_player_season_dataset.csv",index=False)
    descriptions=pd.DataFrame({"feature":BASE_FEATURES+HISTORY_FEATURES,"description":["Current end-of-season player statistic"]*len(BASE_FEATURES)+["Leakage-safe prior-career / prior-season feature"]*len(HISTORY_FEATURES)})
    descriptions.to_csv(OUT/"dataset_feature_description.csv",index=False)
    all_metrics=[]; rankings=[]; all_preds=[]
    for label,features in [("Baseline",BASE_FEATURES),("Improved",BASE_FEATURES+HISTORY_FEATURES)]:
        preds, tests=walk_forward(data,features,completed); preds["experiment"]=label
        ranked,ranks=ranking_metrics(preds,label); ranked.to_csv(OUT/f"season_wise_mvp_predictions_{label.lower()}.csv",index=False)
        ranked[ranked.predicted_rank<=5].to_csv(OUT/f"top_5_candidates_{label.lower()}.csv",index=False)
        all_metrics.append(metric_rows(preds,label)); rankings.append(ranks); all_preds.append(ranked); plot_curves(preds,label)
    metrics=pd.concat(all_metrics); rank=pd.concat(rankings)
    # Threshold 0.50 comparison plus ranking quality; all thresholds remain separately auditable.
    comparison=metrics[metrics.Threshold.eq(.50)].merge(rank.rename(columns={"model":"Model"}),on=["Model","Experiment"])
    comparison.to_csv(OUT/"model_comparison.csv",index=False); metrics.to_csv(OUT/"threshold_metrics.csv",index=False)
    combined=pd.concat(all_preds); combined.to_csv(OUT/"season_wise_mvp_predictions.csv",index=False)
    # Experimental voting-share regression, evaluated only on same completed OOF seasons.
    regrows=[]
    for s in sorted(set(completed)):
        if sum(x<s for x in completed)<8: continue
        train=data[(data.season.isin(completed))&(data.season<s)]; test=data[(data.season==s)]
        pipe=Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("scaler",StandardScaler()),("model",Ridge(alpha=10))])
        pipe.fit(train[BASE_FEATURES+HISTORY_FEATURES],train.voting_share); z=np.clip(pipe.predict(test[BASE_FEATURES+HISTORY_FEATURES]),0,None)
        regrows.append(pd.DataFrame({"season":s,"player_id":test.player_id,"player":test.player,"actual_voting_share":test.voting_share,"predicted_voting_share":z}))
    pd.concat(regrows).to_csv(OUT/"voting_share_regression_predictions.csv",index=False)
    feature_importance(data,BASE_FEATURES+HISTORY_FEATURES,completed)
    pd.DataFrame({"check":["duplicate player-season keys","rookies retained","completed award seasons","evaluation test seasons"],"value":[int(data.duplicated(["season","player_id"]).sum()),int(data.is_rookie.sum()),", ".join(map(str,completed)),", ".join(map(str,tests))]}).to_csv(OUT/"verification_checks.csv",index=False)
    print("Completed MVP seasons:", min(completed), "to", max(completed), "| Walk-forward test seasons:", min(tests), "to", max(tests))
    print(comparison[["Model","Experiment","Precision","Recall","F1","PR-AUC","Top-1 accuracy","Top-3 accuracy","Top-5 accuracy"]].to_string(index=False))

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

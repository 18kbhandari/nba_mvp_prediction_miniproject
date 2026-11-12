"""Streamlit UI for the NBA MVP Prediction mini-project.

Run from the repository root:
    streamlit run app.py

Pages read the CSVs the notebook writes to outputs/. The "Try your own stats"
page re-trains the same four pipelines as the notebook (seasons <= 2021).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier  # noqa: F401  (kept for parity with notebook)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

OUT = Path(__file__).parent / "outputs"
SEED = 42
TRAIN_END = 2021

FEATURES = [
    "age", "g", "gs", "mp_per_game", "fg_percent", "x3p_percent", "ft_percent",
    "trb_per_game", "ast_per_game", "stl_per_game", "blk_per_game",
    "tov_per_game", "pts_per_game",
]
LABELS = {
    "age": "Age", "g": "Games played", "gs": "Games started",
    "mp_per_game": "Minutes / game", "fg_percent": "FG %", "x3p_percent": "3P %",
    "ft_percent": "FT %", "trb_per_game": "Rebounds / game",
    "ast_per_game": "Assists / game", "stl_per_game": "Steals / game",
    "blk_per_game": "Blocks / game", "tov_per_game": "Turnovers / game",
    "pts_per_game": "Points / game",
}
PROB_COLS = {
    "Logistic Regression": "Logistic_Probability",
    "Decision Tree": "DecisionTree_Probability",
    "Random Forest": "RandomForest_Probability",
    "SVM": "SVM_Probability",
}
GOLD, GREY = "#F2A900", "#5B6B7F"

st.set_page_config(page_title="NBA MVP Predictor", page_icon="🏀", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name)


@st.cache_resource(show_spinner="Training the four models (one-time, ~1 min)…")
def train_models():
    data = load_csv("nba_mvp_model_data.csv")
    train = data[data["season"] <= TRAIN_END]
    X, y = train[FEATURES], train["MVP"].astype(int)
    imp = lambda: ("imputer", SimpleImputer(strategy="median"))
    sc = lambda: ("scaler", StandardScaler())
    models = {
        "Logistic Regression": Pipeline([imp(), sc(), ("model", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED))]),
        "Decision Tree": Pipeline([imp(), ("model", DecisionTreeClassifier(
            class_weight="balanced", max_depth=5, random_state=SEED))]),
        "Random Forest": Pipeline([imp(), ("model", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", max_depth=8,
            random_state=SEED, n_jobs=-1))]),
        "SVM": Pipeline([imp(), sc(), ("model", SVC(
            class_weight="balanced", probability=True, kernel="rbf", random_state=SEED))]),
    }
    for m in models.values():
        m.fit(X, y)
    return models, train[FEATURES].median()


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🏀 NBA MVP Predictor")
page = st.sidebar.radio(
    "Go to",
    ["Season rankings", "Model comparison", "Feature importance", "Try your own stats", "About"],
)
st.sidebar.caption(
    "Historical, end-of-season classification. Trained on seasons ≤ 2021, "
    "tested on 2022–2025."
)


# ------------------------------------------------------------------ pages
def page_rankings():
    st.title("Season MVP rankings")
    st.write("Held-out seasons (2022–2025). Players are ranked within each season by the "
             "**combined score** — the average of the four models' probabilities.")
    ranking = load_csv("final_mvp_ranking.csv")
    season_eval = load_csv("season_ranking_evaluation.csv")

    c1, c2 = st.columns([1, 1])
    season = c1.selectbox("Season", sorted(ranking["season"].unique(), reverse=True))
    top_n = c2.slider("Show top N candidates", 3, 15, 8)

    df = ranking[ranking["season"] == season].sort_values("Season_Rank")
    top = df.head(top_n).copy()
    actual = df[df["MVP"] == 1].iloc[0]
    pred = df.iloc[0]
    actual_rank = int(actual["Season_Rank"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Model's #1 pick", pred["player"], f"score {pred['Combined_MVP_Score']:.2f}", delta_color="off")
    m2.metric("Actual MVP", actual["player"])
    m3.metric("Actual MVP's model rank", f"#{actual_rank}",
              "✅ correct" if actual_rank == 1 else ("in top 5" if actual_rank <= 5 else "outside top 5"),
              delta_color="normal" if actual_rank == 1 else "off")

    top["Result"] = np.where(top["MVP"] == 1, "Actual MVP", "Other")
    fig = px.bar(
        top.iloc[::-1], x="Combined_MVP_Score", y="player", orientation="h",
        color="Result", color_discrete_map={"Actual MVP": GOLD, "Other": GREY},
        hover_data={"pts_per_game": True, "trb_per_game": True, "ast_per_game": True,
                    "Season_Rank": True, "Result": False},
        labels={"Combined_MVP_Score": "Combined MVP score", "player": ""},
        category_orders={"player": list(top["player"][::-1])},
    )
    fig.update_layout(height=max(320, 42 * top_n), legend_title_text="", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Per-model probabilities")
    show = top[["Season_Rank", "player", "MVP", *PROB_COLS.values(), "Combined_MVP_Score"]].rename(
        columns={"Season_Rank": "Rank", "player": "Player", "MVP": "Won MVP",
                 **{v: k for k, v in PROB_COLS.items()}, "Combined_MVP_Score": "Combined"})
    show["Won MVP"] = show["Won MVP"].map({1: "🏆", 0: ""})
    prob_cfg = {c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="%.2f")
                for c in [*PROB_COLS.keys(), "Combined"]}
    st.dataframe(show, hide_index=True, width="stretch", column_config=prob_cfg)

    with st.expander("Stat lines of the top candidates"):
        st.dataframe(
            top[["Season_Rank", "player", "age", "g", "mp_per_game", "pts_per_game",
                 "trb_per_game", "ast_per_game", "stl_per_game", "blk_per_game",
                 "fg_percent", "x3p_percent", "ft_percent"]],
            hide_index=True, width="stretch")

    st.subheader("All held-out seasons at a glance")
    ev = season_eval.copy()
    ev["Top-1 correct"] = ev["Top-1 correct"].map({1: "✅", 0: "❌"})
    ev["MVP in top 5"] = ev["MVP in top 5"].map({1: "✅", 0: "❌"})
    st.dataframe(ev, hide_index=True, width="stretch")


def page_models():
    st.title("Model comparison")
    ev = load_csv("model_evaluation.csv")
    metrics = ["Precision", "Recall", "F1-Score", "ROC-AUC", "Average Precision"]

    models_only = ev[ev["Model"] != "Majority baseline"]
    best = models_only.sort_values("Average Precision", ascending=False).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model (by Average Precision)", best["Model"])
    c2.metric("Its Average Precision", f"{best['Average Precision']:.3f}")
    c3.metric("Its ROC-AUC", f"{best['ROC-AUC']:.3f}")

    sel = st.multiselect("Metrics to plot", metrics, default=metrics)
    long = ev.melt(id_vars="Model", value_vars=sel, var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group")
    fig.update_layout(yaxis_range=[0, 1.05], height=430, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        ev.style.format({c: "{:.3f}" for c in ev.columns if c != "Model"}),
        hide_index=True, width="stretch")

    st.info(
        "**Why accuracy is misleading here:** only ~0.2% of player-seasons are MVPs, so the "
        "*Majority baseline* gets 99.8% accuracy while never finding a single MVP (recall 0). "
        "Average Precision and ROC-AUC are the fairer metrics for this imbalanced problem."
    )


def page_importance():
    st.title("Feature importance")
    files = {
        "Random Forest": "random_forest_feature_importance.csv",
        "Decision Tree": "decision_tree_feature_importance.csv",
        "Logistic Regression": "logistic_regression_feature_importance.csv",
        "SVM": "svm_feature_importance.csv",
    }
    notes = {
        "Random Forest": "Mean decrease in impurity.",
        "Decision Tree": "Mean decrease in impurity.",
        "Logistic Regression": "Absolute standardized coefficients.",
        "SVM": "Permutation importance on the test set (drop in average precision).",
    }
    tabs = st.tabs(list(files))
    for tab, (name, f) in zip(tabs, files.items()):
        with tab:
            if (OUT / f).exists():
                d = load_csv(f)
            else:  # file not exported by the notebook -> derive from the trained model
                est = train_models()[0][name].named_steps["model"]
                d = pd.DataFrame({"Feature": FEATURES, "Importance": est.feature_importances_})
            d["Feature"] = d["Feature"].map(LABELS).fillna(d["Feature"])
            d = d.sort_values("Importance")
            fig = px.bar(d, x="Importance", y="Feature", orientation="h",
                         color_discrete_sequence=[GOLD])
            fig.update_layout(height=470, margin=dict(t=10, l=0))
            st.plotly_chart(fig, width="stretch")
            st.caption(notes[name])


def page_predict():
    st.title("Try your own stats")
    st.write("Enter a hypothetical season stat line and see how each model scores it. "
             "Presets are real MVP-caliber lines; adjust any slider.")
    models, medians = train_models()

    presets = {
        "Superstar scorer": dict(age=27, g=70, gs=70, mp_per_game=34, fg_percent=.52, x3p_percent=.36,
                                 ft_percent=.85, trb_per_game=8, ast_per_game=6, stl_per_game=1.3,
                                 blk_per_game=0.8, tov_per_game=3.5, pts_per_game=30),
        "Dominant big man": dict(age=26, g=74, gs=74, mp_per_game=33.5, fg_percent=.58, x3p_percent=.34,
                                 ft_percent=.81, trb_per_game=13.5, ast_per_game=8, stl_per_game=1.5,
                                 blk_per_game=0.9, tov_per_game=3.8, pts_per_game=27),
        "Solid starter": dict(age=26, g=75, gs=75, mp_per_game=30, fg_percent=.46, x3p_percent=.36,
                              ft_percent=.80, trb_per_game=5, ast_per_game=4, stl_per_game=1.0,
                              blk_per_game=0.5, tov_per_game=2, pts_per_game=16),
        "Bench player": dict(age=24, g=60, gs=5, mp_per_game=15, fg_percent=.43, x3p_percent=.33,
                             ft_percent=.72, trb_per_game=3, ast_per_game=1.5, stl_per_game=0.5,
                             blk_per_game=0.3, tov_per_game=1, pts_per_game=6),
    }
    choice = st.selectbox("Start from a preset", list(presets))
    p = presets[choice]

    rng = {
        "age": (19, 42, 1), "g": (1, 82, 1), "gs": (0, 82, 1), "mp_per_game": (5.0, 42.0, 0.5),
        "fg_percent": (0.30, 0.70, 0.005), "x3p_percent": (0.0, 0.55, 0.005),
        "ft_percent": (0.40, 1.0, 0.005), "trb_per_game": (0.0, 18.0, 0.1),
        "ast_per_game": (0.0, 14.0, 0.1), "stl_per_game": (0.0, 3.5, 0.1),
        "blk_per_game": (0.0, 4.0, 0.1), "tov_per_game": (0.0, 6.0, 0.1),
        "pts_per_game": (0.0, 40.0, 0.1),
    }
    vals = {}
    cols = st.columns(3)
    for i, f in enumerate(FEATURES):
        lo, hi, step = rng[f]
        is_int = isinstance(step, int)
        default = type(lo)(p[f]) if not is_int else int(p[f])
        with cols[i % 3]:
            vals[f] = st.slider(LABELS[f], lo, hi, default, step, key=f"{choice}_{f}")

    X = pd.DataFrame([vals])[FEATURES]
    probs = {name: float(m.predict_proba(X)[0, 1]) for name, m in models.items()}
    combined = float(np.mean(list(probs.values())))

    st.divider()
    st.metric("Combined MVP score", f"{combined:.2f}")
    fig = go.Figure(go.Bar(
        x=list(probs.values()), y=list(probs.keys()), orientation="h",
        marker_color=GOLD, text=[f"{v:.2f}" for v in probs.values()], textposition="outside"))
    fig.update_layout(xaxis_range=[0, 1.1], height=260, margin=dict(t=10, l=0),
                      xaxis_title="Model probability")
    st.plotly_chart(fig, width="stretch")
    st.caption("These are ranking scores from class-balanced models, not calibrated probabilities. "
               "Read them as 'how MVP-like is this stat line', not 'chance of winning'.")


def page_about():
    st.title("About this project")
    st.markdown(
        """
**Goal:** predict which player wins the NBA MVP from season-end per-game statistics.

**Data:** *Player Per Game* and *Player Award Shares* (Basketball-Reference style CSVs),
joined into one player-season table. Target = 1 for the MVP winner, 0 otherwise.

**Models:** Logistic Regression, Decision Tree, Random Forest, SVM (RBF) — plus a
majority-class baseline. Imputation and scaling are fitted inside sklearn pipelines on
training data only.

**Evaluation:** chronological split — train through 2021, test on 2022–2025. Metrics:
accuracy, precision, recall, F1, ROC-AUC, average precision, plus season-level ranking.

**Limitations**
- Heavy class imbalance (MVPs are ~0.2% of rows).
- Only four test seasons, one MVP each. The combined ranking puts the true MVP first in
  2 of 4 seasons and in the top 5 in all 4 — preliminary, not proof of reliability.
- Features are season-end stats, so this is a retrospective ranking, not a pre-season forecast.
- The combined score is a ranking score, not a calibrated probability.
        """
    )


{
    "Season rankings": page_rankings,
    "Model comparison": page_models,
    "Feature importance": page_importance,
    "Try your own stats": page_predict,
    "About": page_about,
}[page]()

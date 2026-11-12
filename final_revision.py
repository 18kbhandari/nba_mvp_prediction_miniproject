"""Controlled final MVP experiments: recency, features, candidates, and four models."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from run_analysis import clean_and_engineer, BASE_FEATURES

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True); SEED=42
B_HISTORY=['seasons_played','is_rookie','previous_season_points','previous_season_assists','previous_season_rebounds','career_avg_points','last_3_seasons_avg_points']
FEATURES={'A_original_13':BASE_FEATURES,'B_history':BASE_FEATURES+B_HISTORY}
VALIDATION=list(range(2018,2025)); HOLDOUT=2025

def models(y):
    ratio=max(1,(len(y)-int(y.sum()))/max(1,int(y.sum())))
    return {
      'Logistic Regression':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),('model',LogisticRegression(class_weight='balanced',max_iter=1500,random_state=SEED))]),
      'SVM (RBF)':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),('model',SVC(kernel='rbf',class_weight='balanced',probability=True,random_state=SEED))]),
      'Random Forest':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('model',RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=3,min_samples_split=8,class_weight='balanced',n_jobs=-1,random_state=SEED))]),
      'XGBoost':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('model',XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.05,subsample=.85,colsample_bytree=.85,scale_pos_weight=ratio,eval_metric='logloss',random_state=SEED,n_jobs=2))])}

def candidate_pool(frame, population):
    """Current-season stats only; never uses awards, shares, or labels."""
    x=frame.copy(); x['_score']=x[['pts_per_game','ast_per_game','trb_per_game']].fillna(0).sum(axis=1)
    if population=='All': return x.drop(columns='_score')
    n=int(population.replace('Top',''))
    return x.sort_values(['season','_score','player_id'],ascending=[True,False,True]).groupby('season',group_keys=False).head(n).drop(columns='_score')

def train_window(train, test_year, window):
    return train if window=='Full' else train[train.season >= test_year-int(window)]

def evaluate(data, completed, features, window='Full', population='All', seasons=VALIDATION):
    out=[]
    for year in seasons:
        train=data[(data.season.isin(completed))&(data.season<year)].copy()
        train=train_window(train,year,window); test=data[data.season.eq(year)].copy()
        train=candidate_pool(train,population); test=candidate_pool(test,population)
        included=int(test.MVP.sum())
        for name,pipe in models(train.MVP).items():
            pipe.fit(train[features],train.MVP); z=pipe.predict_proba(test[features])[:,1]
            part=test[['season','player_id','player','team','MVP','voting_share']].copy(); part['model']=name; part['probability']=z; part['window']=window; part['population']=population; part['mvp_in_candidate_pool']=included
            out.append(part)
    return pd.concat(out,ignore_index=True)

def summarize(preds, label, setting, value):
    rows=[]; ranked=preds.copy(); ranked['predicted_rank']=ranked.groupby(['model','season']).probability.rank(method='first',ascending=False).astype(int)
    for model,x in ranked.groupby('model'):
        y,s=x.MVP,x.probability; z=(s>=.5).astype(int); cm=confusion_matrix(y,z,labels=[0,1])
        winners=x[x.MVP.eq(1)]; included=x.groupby('season').mvp_in_candidate_pool.first().mean()
        rows.append({'Experiment':label,'Setting':setting,'Value':value,'Model':model,'Validation seasons':f'{min(x.season)}-{max(x.season)}','Seasons':x.season.nunique(),'Candidate rows':len(x),'Candidates/season':len(x)/x.season.nunique(),'MVP inclusion rate':included,'Top-1 accuracy':(winners.predicted_rank<=1).mean() if len(winners) else np.nan,'Top-3 accuracy':(winners.predicted_rank<=3).mean() if len(winners) else np.nan,'Top-5 accuracy':(winners.predicted_rank<=5).mean() if len(winners) else np.nan,'Accuracy':accuracy_score(y,z),'Precision':precision_score(y,z,zero_division=0),'Recall':recall_score(y,z,zero_division=0),'F1':f1_score(y,z,zero_division=0),'ROC-AUC':roc_auc_score(y,s) if y.nunique()==2 else np.nan,'PR-AUC':average_precision_score(y,s),'False positives':int(cm[0,1]),'True positives':int(cm[1,1])})
    return pd.DataFrame(rows),ranked

def choose(summary):
    # Ranking is primary; PR-AUC and precision break ties without using holdout data.
    return summary.sort_values(['Top-1 accuracy','Top-3 accuracy','PR-AUC','Precision'],ascending=False).iloc[0]

def plot_final(preds, name):
    fig,ax=plt.subplots(figsize=(7,5))
    for model,x in preds.groupby('model'):
        p,r,_=precision_recall_curve(x.MVP,x.probability); ax.plot(r,p,label=model)
    ax.set(title='Final validation precision-recall curves',xlabel='Recall',ylabel='Precision'); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT/'final_precision_recall_curves.png',dpi=160); plt.close(fig)
    x=preds[preds.model.eq(name)]; z=(x.probability>=.5).astype(int); cm=confusion_matrix(x.MVP,z,labels=[0,1])
    fig,ax=plt.subplots(figsize=(4,3)); ax.imshow(cm,cmap='Blues'); ax.set(title=f'{name}: validation confusion matrix',xlabel='Predicted',ylabel='Actual',xticks=[0,1],yticks=[0,1]);
    for i in range(2):
      for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.tight_layout(); fig.savefig(OUT/'final_confusion_matrix.png',dpi=160); plt.close(fig)

def final_importance(data,completed,features,window,model_name):
    train=data[(data.season.isin(completed))&(data.season<HOLDOUT)]; train=train_window(train,HOLDOUT,window)
    pipe=models(train.MVP)[model_name]; pipe.fit(train[features],train.MVP)
    test=data[data.season.eq(HOLDOUT)]
    pi=permutation_importance(pipe,test[features],test.MVP,n_repeats=8,random_state=SEED,scoring='average_precision')
    x=pd.DataFrame({'Feature':features,'Permutation importance':pi.importances_mean}).sort_values('Permutation importance',ascending=False)
    x.to_csv(OUT/'final_feature_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,6)); q=x.head(15).sort_values('Permutation importance'); ax.barh(q.Feature,q['Permutation importance']); ax.set_title(f'{model_name}: holdout permutation importance'); fig.tight_layout(); fig.savefig(OUT/'final_feature_importance.png',dpi=160); plt.close(fig)

def main():
    data,completed=clean_and_engineer(); completed=[s for s in completed if s<=HOLDOUT]
    data=data[data.season<=HOLDOUT].copy(); data.to_csv(OUT/'final_revision_cleaned_dataset.csv',index=False)
    # Controlled 1: recency windows; fixed A and all candidates.
    tables=[]
    for w in [10,15,20,25,'Full']:
        s,r=summarize(evaluate(data,completed,FEATURES['A_original_13'],w,'All'),'Training window','Window',str(w)); tables.append(s)
    windows=pd.concat(tables,ignore_index=True); windows.to_csv(OUT/'training_window_comparison.csv',index=False)
    selected_window=choose(windows)['Value']
    selected_window=int(selected_window) if selected_window!='Full' else 'Full'
    # Controlled 2: feature sets. Team data is absent, so C is documented unavailable instead of fabricated.
    ft=[]
    for name,cols in FEATURES.items():
        s,r=summarize(evaluate(data,completed,cols,selected_window,'All'),'Feature configuration','Features',name); ft.append(s)
    feature_table=pd.concat(ft,ignore_index=True); feature_table.to_csv(OUT/'feature_experiment_comparison.csv',index=False)
    pd.DataFrame([{'Configuration':'C_team_features','Status':'Not run: repository has no reliable team wins, win percentage, or standings data; B is not duplicated as C.','Features':'team_wins, team_win_percentage, conference_standing'}]).to_csv(OUT/'team_feature_availability.csv',index=False)
    selected_feature=choose(feature_table)['Value']; selected_features=FEATURES[selected_feature]
    # Controlled 3: candidate population.
    ct=[]
    for pop in ['All','Top10','Top5']:
        s,r=summarize(evaluate(data,completed,selected_features,selected_window,pop),'Candidate population','Population',pop); ct.append(s)
    candidate_table=pd.concat(ct,ignore_index=True); candidate_table.to_csv(OUT/'candidate_population_comparison.csv',index=False)
    # Pick methodology from validation; models remain comparable under identical folds.
    all_control=pd.concat([windows,feature_table,candidate_table],ignore_index=True)
    # Final common comparison uses the best population by average Top-1 across models, not a model-specific winner.
    population_scores=candidate_table.groupby('Value')['Top-1 accuracy'].mean().sort_values(ascending=False); selected_population=population_scores.index[0]
    final_preds=evaluate(data,completed,selected_features,selected_window,selected_population)
    final_summary,ranked=summarize(final_preds,'Final controlled comparison','Configuration',f'window={selected_window}; features={selected_feature}; candidates={selected_population}')
    final_summary.to_csv(OUT/'model_performance_comparison.csv',index=False); ranked.to_csv(OUT/'season_wise_predictions_final_validation.csv',index=False)
    best=choose(final_summary); best_model=best['Model']; plot_final(final_preds,best_model)
    # Untouched 2025 final evaluation: only after all selection above.
    hold=evaluate(data,completed,selected_features,selected_window,selected_population,seasons=[HOLDOUT]); hold_summary,hold_rank=summarize(hold,'Untouched holdout','Configuration',f'window={selected_window}; features={selected_feature}; candidates={selected_population}')
    hold_summary.to_csv(OUT/'final_holdout_evaluation.csv',index=False); hold_rank.to_csv(OUT/'season_wise_predictions_holdout_2025.csv',index=False)
    final_importance(data,completed,selected_features,selected_window,best_model)
    pd.DataFrame([{'selected_training_window':selected_window,'selected_features':selected_feature,'selected_candidate_population':selected_population,'selected_model':best_model,'selection_rule':'highest validation Top-1; then Top-3, PR-AUC, precision','validation_seasons':'2018-2024','untouched_holdout':HOLDOUT}]).to_csv(OUT/'final_model_selection.csv',index=False)
    print(final_summary[['Model','Top-1 accuracy','Top-3 accuracy','Precision','PR-AUC']].to_string(index=False)); print('Selected:',best_model,selected_window,selected_feature,selected_population)

if __name__=='__main__': main()

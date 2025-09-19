

"""
Advanced end-to-end Data Science project
Dataset: sklearn's load_wine (multiclass classification) -- replaceable with your CSV.
Features:
 - EDA (summary)
 - Preprocessing (scaling, imputation)
 - Feature engineering (polynomial interactions example + tree-based selection)
 - Models: RandomForest, XGBoost (if available), LightGBM (if available), LogisticRegression
 - Stacking with meta-learner
 - StratifiedKFold CV with cross_val_predict style stacking
 - RandomizedSearch for hyperparameter tuning
 - SHAP interpretation (if installed)
 - Save final model with joblib
"""



import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.base import clone
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns


try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

try:
    import shap
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False


data = load_wine(as_frame=True)
X = data.data.copy()
y = data.target.copy()
feature_names = X.columns.tolist()


print("Dataset shape:", X.shape)
print("Classes distribution:\n", y.value_counts(normalize=True))
print("\nFeature head:\n", X.head()
def quick_plots():
    plt.figure(figsize=(10,6))
    sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
    plt.title("Feature correlation matrix")
    plt.show()


numeric_features = feature_names

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features)
])


def add_interactions(df, top_k=5):
    corr = df.corr().abs().mean().sort_values(ascending=False)
    top_feats = corr.index[:top_k].tolist()
    df2 = df.copy()
    for i in range(len(top_feats)):
        for j in range(i+1, len(top_feats)):
            a, b = top_feats[i], top_feats[j]
            df2[f"{a}_x_{b}"] = df2[a] * df2[b]
    return df2

X_fe = add_interactions(X, top_k=5)
print("After interactions shape:", X_fe.shape)

all_features = X_fe.columns.tolist()


numeric_transformer_fe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])
preprocessor_fe = ColumnTransformer(transformers=[
    ('num', numeric_transformer_fe, all_features)
])


base_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

pipeline_rf = Pipeline(steps=[
    ('preproc', preprocessor_fe),
    ('feature_sel', SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=42), threshold="median")),
    ('clf', base_clf)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_scores = cross_val_score(pipeline_rf, X_fe, y, cv=cv, scoring='accuracy', n_jobs=-1)
print("RandomForest CV accuracy:", np.round(acc_scores.mean(),4), "±", np.round(acc_scores.std(),4))


pipeline_rf.fit(X_fe, y)


estimators = []


estimators.append(('rf', clone(base_clf)))


estimators.append(('lr', LogisticRegression(max_iter=200)))


if HAVE_XGB:
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
    estimators.append(('xgb', xgb_clf))
else:
    print("xgboost not available — skipping XGB in ensemble.")


if HAVE_LGB:
    lgb_clf = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
    estimators.append(('lgb', lgb_clf))
else:
    print("lightgbm not available — skipping LightGBM in ensemble.")


stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=500),
    cv=cv,
    n_jobs=-1,
    passthrough=False
)

stack_pipeline = Pipeline(steps=[
    ('preproc', preprocessor_fe),
    ('stack', stack)
])


stack_scores = cross_val_score(stack_pipeline, X_fe, y, cv=cv, scoring='accuracy', n_jobs=-1)
print("Stacking CV accuracy:", np.round(stack_scores.mean(),4), "±", np.round(stack_scores.std(),4))


stack_pipeline.fit(X_fe, y)


param_dist = {
    'clf__n_estimators': [100, 200, 400],
    'clf__max_depth': [None, 6, 10, 20],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}


tune_pipe = Pipeline(steps=[
    ('preproc', preprocessor_fe),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

rnd_search = RandomizedSearchCV(
    estimator=tune_pipe,
    param_distributions=param_dist,
    n_iter=12,
    scoring='accuracy',
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rnd_search.fit(X_fe, y)
print("Best tune params:", rnd_search.best_params_)
print("Best tune CV score:", rnd_search.best_sc
final_model = stack_pipeline


from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(final_model, X_fe, y, cv=cv, method='predict', n_jobs=-1)
print("\nClassification report (cv predictions):\n", classification_report(y, y_pred))
print("Confusion matrix:\n", confusion_matrix(y, y_pred))


if HAVE_SHAP:

    X_trans = preprocessor_fe.fit_transform(X_fe)  
    rf_in_stack = None
    for name, est in stack.named_estimators_.items():
        if isinstance(est, RandomForestClassifier):
            rf_in_stack = est
            break

    if rf_in_stack is not None:
        explainer = shap.TreeExplainer(rf_in_stack)
        shap_values = explainer.shap_values(X_trans)
        
        print("Displaying SHAP summary plot for class 0 (if running in notebook).")
        shap.summary_plot(shap_values[0], features=X_trans, feature_names=all_features)
    else:
        print("No RF in stack to explain with SHAP easily.")
else:
    print("shap not installed — skip interpretation. pip install shap to enable.")

dump(final_model, "final_stacked_model.joblib")
dump(preprocessor_fe, "preprocessor_fe.joblib")
print("Saved final model to final_stacked_model.joblib and preprocessor_fe.joblib")


def predict_single(sample_dict):
    """
    sample_dict: dict of feature_name -> value (must include interaction features or we'll recompute)
    We'll compute interactions same way as training (safe approach: build a DF)
    """
    df = pd.DataFrame([sample_dict])
    
    for col in X.columns:
        if col not in df.columns:
            df[col] = np.nan
    
    df_fe = add_interactions(df, top_k=5)
    
    df_fe = df_fe.reindex(columns=all_features, fill_value=np.nan)
    preds = final_model.predict(df_fe)
    probs = final_model.predict_proba(df_fe)
    return preds[0], probs[0]


sample = X.mean().to_dict()
pred_class, pred_proba = predict_single(sample)
print("Sample prediction class:", pred_class, "proba:", pred_proba)

# --------------------------
# 12) Tips to adapt to real dataset:
# - Replace load_wine() with pd.read_csv('yourfile.csv')
# - Handle categorical features: use OneHotEncoder or TargetEncoder accordingly
# - If dataset large: consider incremental learning or sampling strategies
# - For imbalance: use class_weight or oversampling (SMOTE)
# - For production: package preprocessing + model into a single pipeline and export with joblib or ONNX
# --------------------------


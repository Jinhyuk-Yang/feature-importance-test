# 목적: LightGBM 중요도/해석 결과를 PNG로 저장

import os
import matplotlib
from lightgbm import LGBMClassifier
from pathlib import Path
from sklearn.metrics import roc_auc_score

from utils.make_dataset import make_classification_dataset
from models.feature_importance import FeatureImportanceExplainer

matplotlib.use("Agg")

OUTDIR = Path("artifacts/classification")
os.makedirs(OUTDIR, exist_ok=True)

# 분류 데이터셋 랜덤 생성
n_samples = 8000
n_classes = 2
n_features = 25
feature_names = [f"f_{i:02d}" for i in range(n_features)]
X_tr, X_te, y_tr, y_te = make_classification_dataset(
    n_samples=n_samples,
    n_classes=n_classes,
    n_features=n_features,
    feature_names=feature_names,
)

# LightGBM 학습
model = LGBMClassifier(
    n_estimators=600,
    num_leaves=63,
    learning_rate=0.03,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric="auc")
auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
print(f"학습 완료! AUC: {auc:.4f}")

# Feature 분석기 가동
feature_explainer = FeatureImportanceExplainer(model, feature_names, OUTDIR)

# Feature Importance & Permutation Importance 체크
feature_explainer.get_feature_importance()
print("[info] feature importance 결과 CSV/PNG 저장 완료")
feature_explainer.get_permutation_importance(X_te, y_te)
print("[info] permutation importance 결과 CSV/PNG 저장 완료")

# SHAP 체크
feature_explainer.get_shap(X_te, n_sampling=1000)
print("[info] SHAP 결과 CSV/PNG 저장 완료")

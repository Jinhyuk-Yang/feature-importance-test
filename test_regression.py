# 목적: LightGBM 회귀모델로 연속값 예측 + feature importance 저장/시각화

import os
import matplotlib
import numpy as np
from lightgbm import LGBMRegressor
from pathlib import Path
from sklearn.metrics import mean_squared_error, make_scorer

from utils.make_dataset import make_regression_dataset
from models.feature_importance import FeatureImportanceExplainer

matplotlib.use("Agg")


def rmse_func(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


OUTDIR = Path("artifacts/regression")
os.makedirs(OUTDIR, exist_ok=True)

# 회귀 데이터셋 랜덤 생성
n_samples = 8000
n_features = 25
feature_names = [f"f_{i:02d}" for i in range(n_features)]
X_tr, X_te, y_tr, y_te = make_regression_dataset(
    n_samples=n_samples,
    n_features=n_features,
    feature_names=feature_names,
)

# LightGBM 학습
model = LGBMRegressor(
    n_estimators=500,
    num_leaves=63,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric="l2")
rmse = rmse_func(y_te, model.predict(X_te))
print(f"학습 완료! RMSE: {rmse:.4f}")

# Feature 분석기 가동
feature_explainer = FeatureImportanceExplainer(model, feature_names, OUTDIR)

# Feature Importance & Permutation Importance 체크
feature_explainer.get_feature_importance()
print("[info] feature importance 결과 CSV/PNG 저장 완료")
feature_explainer.get_permutation_importance(
    X_te, y_te, topN=15, scoring=make_scorer(rmse_func, greater_is_better=False)
)
print("[info] permutation importance 결과 CSV/PNG 저장 완료")

# SHAP 체크
feature_explainer.get_shap(X_te, n_sampling=1000)
print("[info] SHAP 결과 CSV/PNG 저장 완료")

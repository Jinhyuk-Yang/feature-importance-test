import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier, LGBMRegressor
from pathlib import Path
from sklearn.inspection import permutation_importance
from typing import Any

from utils.save_file import save_dataframe_as_csv
from utils.plotters import (
    plot_feature_importance,
    plot_permutation_importance,
    plot_shap_summary,
    plot_shap_dependence,
)


class FeatureImportanceExplainer:
    def __init__(
        self,
        model: LGBMClassifier | LGBMRegressor,
        feature_names: list[str],
        output_dir: Path,
    ):
        self.model = model
        self.feature_names = feature_names
        self.output_dir = output_dir

    def get_feature_importance(self, topN: int = 20) -> pd.DataFrame:
        imp_split = pd.Series(
            self.model.booster_.feature_importance(importance_type="split"),
            index=self.feature_names,
            name="split_count",
        )
        imp_gain = pd.Series(
            self.model.booster_.feature_importance(importance_type="gain"),
            index=self.feature_names,
            name="total_gain",
        )
        imp_df = pd.concat([imp_split, imp_gain], axis=1).sort_values(
            "total_gain", ascending=False
        )
        save_dataframe_as_csv(imp_df, self.output_dir / "lgbm_feature_importance.csv")

        top_gain = imp_df.sort_values("total_gain", ascending=True).tail(topN)
        top_split = imp_df.sort_values("split_count", ascending=True).tail(topN)
        plot_feature_importance(
            top_gain,
            top_split,
            self.output_dir / f"lgbm_feature_importance_top{topN}.png",
        )

    def get_permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        topN: int = 20,
        n_repeats: int = 5,
        scoring: Any = "roc_auc",
    ):
        perm = permutation_importance(
            self.model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=42,
            scoring=scoring,
            n_jobs=-1,
        )
        perm_df = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "imp_mean": perm.importances_mean,
                    "imp_std": perm.importances_std,
                }
            )
            .sort_values("imp_mean", ascending=False)
            .reset_index(drop=True)
        )
        save_dataframe_as_csv(perm_df, self.output_dir / "permutation_importance.csv")

        top_perm = perm_df.head(topN).iloc[::-1]
        plot_permutation_importance(
            top_perm, self.output_dir / f"permutation_importance_top{topN}.png"
        )

    def get_shap(self, X: pd.DataFrame, n_sampling: int = 1000):
        X_sample = X.sample(n=min(n_sampling, len(X)), random_state=42)

        explainer = shap.TreeExplainer(self.model, feature_names=self.feature_names)
        sv = explainer.shap_values(X_sample)

        # summary plot
        plot_shap_summary(sv, X_sample, self.output_dir / "shap_summary.png")

        # dependence plot
        mean_abs = np.abs(sv).mean(axis=0)
        top_feat = X_sample.columns[np.argsort(-mean_abs)][0]  # 가장 영향 큰 피처 선택
        plot_shap_dependence(
            sv,
            X_sample,
            top_feat,
            self.output_dir / f"shap_dependence_{top_feat}.png",
        )

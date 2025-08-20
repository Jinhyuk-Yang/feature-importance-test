import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pathlib import Path


def plot_feature_importance(
    top_gain: pd.DataFrame, top_split: pd.DataFrame, save_path: Path
):
    assert len(top_gain) == len(top_split)
    _, axes = plt.subplots(1, 2, figsize=(16, 0.4 * len(top_gain) + 1))

    axes[0].barh(top_gain.index, top_gain["total_gain"])
    axes[0].set_title("Feature Importance: Gain")
    axes[0].set_xlabel("Total Gain")

    axes[1].barh(top_split.index, top_split["split_count"])
    axes[1].set_title("Feature Importance: Split")
    axes[1].set_xlabel("Split Count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[save] {save_path}")


def plot_permutation_importance(top_perm: pd.DataFrame, save_path: Path):
    plt.figure(figsize=(8, 0.4 * len(top_perm) + 1))
    plt.barh(top_perm["feature"], top_perm["imp_mean"], xerr=top_perm["imp_std"])
    plt.title("Permutation Importance")
    plt.xlabel("Mean Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[save] {save_path}")


def plot_shap_summary(sv: np.ndarray, X_sample: pd.DataFrame, save_path: Path):
    shap.summary_plot(sv, X_sample, show=False)
    plt.title("SHAP Summary (class 1)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[save] {save_path}")


def plot_shap_dependence(
    sv: np.ndarray, X_sample: pd.DataFrame, top_feat: str, save_path: Path
):
    # 보조 상호작용 피처 자동 선택(None) → SHAP가 내부적으로 고름
    shap.dependence_plot(top_feat, sv, X_sample, show=False)
    plt.title(f"SHAP Dependence: {top_feat}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[save] {save_path}")

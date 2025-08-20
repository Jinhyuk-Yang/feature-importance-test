import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


def make_classification_dataset(
    n_samples: int, n_classes: int, n_features: int, feature_names: list[str]
):
    assert n_classes > 1
    assert len(feature_names) == n_features

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=8,
        n_redundant=4,
        random_state=42,
        shuffle=True,
    )

    X = pd.DataFrame(X, columns=feature_names)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_tr, X_te, y_tr, y_te


def make_regression_dataset(n_samples: int, n_features: int, feature_names: list[str]):
    assert len(feature_names) == n_features

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        noise=0.3,
        random_state=42,
    )

    X = pd.DataFrame(X, columns=feature_names)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_tr, X_te, y_tr, y_te

import pandas as pd
from pathlib import Path


def save_dataframe_as_csv(df: pd.DataFrame, save_path: Path):
    df.to_csv(save_path)
    print(f"[save] {save_path}")

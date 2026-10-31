import pandas as pd

# Load and save data
def load_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def save_csv(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    df.to_csv(file_path, index=index)
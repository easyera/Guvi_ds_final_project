import pandas as pd
import re

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop PII columns
    columns_to_drop = ["ticket_id", "customer_name", "customer_email"]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    # Fill browser nulls
    df["browser"] = df["browser"].fillna("Unknown")

    # Parse date columns
    date_columns = ["ticket_created_date", "ticket_resolved_date"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")

    # Clean text columns
    text_columns = df.select_dtypes(include=["object"]).columns
    for col in text_columns:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9\s\.\?]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # Only drop rows where issue_description is empty
    if 'issue_description' in df.columns:
        df = df[df['issue_description'] != ""]

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    return df
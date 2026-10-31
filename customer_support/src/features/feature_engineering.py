import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Parse and extract date features
    df['ticket_created_date'] = pd.to_datetime(df['ticket_created_date'])
    df['day_of_week'] = df['ticket_created_date'].dt.dayofweek
    df['month'] = df['ticket_created_date'].dt.month

    # Drop irrelevant columns
    cols_to_drop = [
        'ticket_created_date',
        'ticket_resolved_date',
        'resolution_notes',
        'issue_complexity_score',
        'status'
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Add ticket rate feature
    df['ticket_rate'] = df['previous_tickets'] / df['customer_tenure_months']

    return df
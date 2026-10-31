import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix, save_npz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


# ── Constants ──────────────────────────────────────────────────────────────────

NUM_COLS_PRIORITY = [
    'customer_age', 'customer_tenure_months', 'previous_tickets',
    'day_of_week', 'month', 'ticket_rate'
]

NUM_COLS_SLA_SATISFACTION = [
    'customer_age', 'customer_tenure_months', 'previous_tickets',
    'first_response_time_hours', 'day_of_week', 'month', 'ticket_rate'
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def encode_cats(X, ohe, cat_cols):
    encoded = ohe.transform(X[cat_cols])
    encoded_df = pd.DataFrame(
        encoded, columns=ohe.get_feature_names_out(cat_cols), index=X.index
    )
    return pd.concat([X.drop(columns=cat_cols), encoded_df], axis=1)


def combine_features(X_df, X_text):
    return hstack([csr_matrix(X_df.values), X_text])


# ── Main ───────────────────────────────────────────────────────────────────────

def run_preprocessing(featured_data_path: str, features_save_dir: str, models_save_dir: str):

    # Load
    df = pd.read_csv(featured_data_path)

    # Separate targets
    y_priority    = df['priority']
    y_sla         = df['sla_breached']
    y_satisfaction = df['customer_satisfaction_score']
    X = df.drop(columns=['priority', 'sla_breached', 'customer_satisfaction_score'])

    # Goal-specific feature sets
    X_priority     = X.drop(columns=['first_response_time_hours', 'escalated'])
    X_sla          = X.copy()
    X_satisfaction = X.copy()

    # Train / val / test split
    def split(X, y):
        X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.176, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    X_priority_train, X_priority_val, X_priority_test, y_priority_train, y_priority_val, y_priority_test = split(X_priority, y_priority)
    X_sla_train, X_sla_val, X_sla_test, y_sla_train, y_sla_val, y_sla_test                             = split(X_sla, y_sla)
    X_satisfaction_train, X_satisfaction_val, X_satisfaction_test, y_satisfaction_train, y_satisfaction_val, y_satisfaction_test = split(X_satisfaction, y_satisfaction)

    # Label encode targets
    le_priority = LabelEncoder()
    y_priority_train = le_priority.fit_transform(y_priority_train)
    y_priority_val   = le_priority.transform(y_priority_val)
    y_priority_test  = le_priority.transform(y_priority_test)

    le_sla = LabelEncoder()
    y_sla_train = le_sla.fit_transform(y_sla_train)
    y_sla_val   = le_sla.transform(y_sla_val)
    y_sla_test  = le_sla.transform(y_sla_test)

    # Categorical columns
    cat_cols_priority          = [c for c in X_priority_train.select_dtypes('object').columns if c != 'issue_description']
    cat_cols_sla_satisfaction  = [c for c in X_sla_train.select_dtypes('object').columns     if c != 'issue_description']

    # One-hot encoding
    ohe_priority = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_priority.fit(X_priority_train[cat_cols_priority])
    X_priority_train = encode_cats(X_priority_train, ohe_priority, cat_cols_priority)
    X_priority_val   = encode_cats(X_priority_val,   ohe_priority, cat_cols_priority)
    X_priority_test  = encode_cats(X_priority_test,  ohe_priority, cat_cols_priority)

    ohe_sla_satisfaction = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_sla_satisfaction.fit(X_sla_train[cat_cols_sla_satisfaction])
    X_sla_train          = encode_cats(X_sla_train,          ohe_sla_satisfaction, cat_cols_sla_satisfaction)
    X_sla_val            = encode_cats(X_sla_val,            ohe_sla_satisfaction, cat_cols_sla_satisfaction)
    X_sla_test           = encode_cats(X_sla_test,           ohe_sla_satisfaction, cat_cols_sla_satisfaction)
    X_satisfaction_train = encode_cats(X_satisfaction_train, ohe_sla_satisfaction, cat_cols_sla_satisfaction)
    X_satisfaction_val   = encode_cats(X_satisfaction_val,   ohe_sla_satisfaction, cat_cols_sla_satisfaction)
    X_satisfaction_test  = encode_cats(X_satisfaction_test,  ohe_sla_satisfaction, cat_cols_sla_satisfaction)

    # Scaling
    scaler_priority = StandardScaler()
    X_priority_train[NUM_COLS_PRIORITY] = scaler_priority.fit_transform(X_priority_train[NUM_COLS_PRIORITY])
    X_priority_val[NUM_COLS_PRIORITY]   = scaler_priority.transform(X_priority_val[NUM_COLS_PRIORITY])
    X_priority_test[NUM_COLS_PRIORITY]  = scaler_priority.transform(X_priority_test[NUM_COLS_PRIORITY])

    scaler_sla = StandardScaler()
    X_sla_train[NUM_COLS_SLA_SATISFACTION] = scaler_sla.fit_transform(X_sla_train[NUM_COLS_SLA_SATISFACTION])
    X_sla_val[NUM_COLS_SLA_SATISFACTION]   = scaler_sla.transform(X_sla_val[NUM_COLS_SLA_SATISFACTION])
    X_sla_test[NUM_COLS_SLA_SATISFACTION]  = scaler_sla.transform(X_sla_test[NUM_COLS_SLA_SATISFACTION])

    scaler_satisfaction = StandardScaler()
    X_satisfaction_train[NUM_COLS_SLA_SATISFACTION] = scaler_satisfaction.fit_transform(X_satisfaction_train[NUM_COLS_SLA_SATISFACTION])
    X_satisfaction_val[NUM_COLS_SLA_SATISFACTION]   = scaler_satisfaction.transform(X_satisfaction_val[NUM_COLS_SLA_SATISFACTION])
    X_satisfaction_test[NUM_COLS_SLA_SATISFACTION]  = scaler_satisfaction.transform(X_satisfaction_test[NUM_COLS_SLA_SATISFACTION])

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    tfidf.fit(X_priority_train['issue_description'].astype(str))

    def tfidf_transform(X):
        return tfidf.transform(X['issue_description'].astype(str))

    X_priority_train_text     = tfidf_transform(X_priority_train)
    X_priority_val_text       = tfidf_transform(X_priority_val)
    X_priority_test_text      = tfidf_transform(X_priority_test)
    X_sla_train_text          = tfidf_transform(X_sla_train)
    X_sla_val_text            = tfidf_transform(X_sla_val)
    X_sla_test_text           = tfidf_transform(X_sla_test)
    X_satisfaction_train_text = tfidf_transform(X_satisfaction_train)
    X_satisfaction_val_text   = tfidf_transform(X_satisfaction_val)
    X_satisfaction_test_text  = tfidf_transform(X_satisfaction_test)

    # Drop issue_description and combine
    for X_df in [X_priority_train, X_priority_val, X_priority_test,
                 X_sla_train, X_sla_val, X_sla_test,
                 X_satisfaction_train, X_satisfaction_val, X_satisfaction_test]:
        X_df.drop(columns=['issue_description'], inplace=True)

    X_priority_train_final     = combine_features(X_priority_train,     X_priority_train_text)
    X_priority_val_final       = combine_features(X_priority_val,       X_priority_val_text)
    X_priority_test_final      = combine_features(X_priority_test,      X_priority_test_text)
    X_sla_train_final          = combine_features(X_sla_train,          X_sla_train_text)
    X_sla_val_final            = combine_features(X_sla_val,            X_sla_val_text)
    X_sla_test_final           = combine_features(X_sla_test,           X_sla_test_text)
    X_satisfaction_train_final = combine_features(X_satisfaction_train, X_satisfaction_train_text)
    X_satisfaction_val_final   = combine_features(X_satisfaction_val,   X_satisfaction_val_text)
    X_satisfaction_test_final  = combine_features(X_satisfaction_test,  X_satisfaction_test_text)

    # Save feature matrices
    save_npz(f'{features_save_dir}/X_priority_train.npz',     X_priority_train_final)
    save_npz(f'{features_save_dir}/X_priority_val.npz',       X_priority_val_final)
    save_npz(f'{features_save_dir}/X_priority_test.npz',      X_priority_test_final)
    save_npz(f'{features_save_dir}/X_sla_train.npz',          X_sla_train_final)
    save_npz(f'{features_save_dir}/X_sla_val.npz',            X_sla_val_final)
    save_npz(f'{features_save_dir}/X_sla_test.npz',           X_sla_test_final)
    save_npz(f'{features_save_dir}/X_satisfaction_train.npz', X_satisfaction_train_final)
    save_npz(f'{features_save_dir}/X_satisfaction_val.npz',   X_satisfaction_val_final)
    save_npz(f'{features_save_dir}/X_satisfaction_test.npz',  X_satisfaction_test_final)

    # Save target arrays
    np.save(f'{features_save_dir}/y_priority_train.npy',     y_priority_train)
    np.save(f'{features_save_dir}/y_priority_val.npy',       y_priority_val)
    np.save(f'{features_save_dir}/y_priority_test.npy',      y_priority_test)
    np.save(f'{features_save_dir}/y_sla_train.npy',          y_sla_train)
    np.save(f'{features_save_dir}/y_sla_val.npy',            y_sla_val)
    np.save(f'{features_save_dir}/y_sla_test.npy',           y_sla_test)
    np.save(f'{features_save_dir}/y_satisfaction_train.npy', y_satisfaction_train)
    np.save(f'{features_save_dir}/y_satisfaction_val.npy',   y_satisfaction_val)
    np.save(f'{features_save_dir}/y_satisfaction_test.npy',  y_satisfaction_test)

    # Save preprocessing artifacts
    joblib.dump(ohe_priority,         f'{models_save_dir}/ohe_priority.pkl')
    joblib.dump(ohe_sla_satisfaction, f'{models_save_dir}/ohe_sla_satisfaction.pkl')
    joblib.dump(scaler_priority,      f'{models_save_dir}/scaler_priority.pkl')
    joblib.dump(scaler_sla,           f'{models_save_dir}/scaler_sla.pkl')
    joblib.dump(scaler_satisfaction,  f'{models_save_dir}/scaler_satisfaction.pkl')
    joblib.dump(tfidf,                f'{models_save_dir}/tfidf.pkl')
    joblib.dump(le_priority,          f'{models_save_dir}/le_priority.pkl')
    joblib.dump(le_sla,               f'{models_save_dir}/le_sla.pkl')

    print("Preprocessing complete.")
    print(f"Priority:     {X_priority_train_final.shape}")
    print(f"SLA:          {X_sla_train_final.shape}")
    print(f"Satisfaction: {X_satisfaction_train_final.shape}")
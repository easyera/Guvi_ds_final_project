import sys
import yaml
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / 'src'))

from data.data_cleaning import clean_data
from features.feature_engineering import engineer_features
from models.priority_classifier import DualInputModel as PriorityModel
from models.sla_classifier import DualInputModel as SLAModel
from models.satisfaction_classifier import DualInputModel as SatisfactionModel


# ── Constants ──────────────────────────────────────────────────────────────────
MAX_LEN    = 13
EMBED_DIM  = 32
HIDDEN_DIM = 64

LEAKY_COLS_PRIORITY = ['first_response_time_hours', 'escalated']
NUM_COLS_PRIORITY   = ['customer_age', 'customer_tenure_months', 'previous_tickets',
                       'day_of_week', 'month', 'ticket_rate']
NUM_COLS_SLA_SAT    = ['customer_age', 'customer_tenure_months', 'previous_tickets',
                       'first_response_time_hours', 'day_of_week', 'month', 'ticket_rate']


def load_config():
    with open(ROOT / 'configs' / 'config.yaml', 'r') as f:
        return yaml.safe_load(f)


# ── Preprocessing helpers ──────────────────────────────────────────────────────

def text_to_sequence(text, vocab):
    return [vocab.get(word, 0) for word in str(text).split()]


def pad_sequence(sequence, max_len):
    return sequence + [0] * (max_len - len(sequence))


def encode_cats(X, ohe, cat_cols):
    encoded    = ohe.transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols), index=X.index)
    return pd.concat([X.drop(columns=cat_cols), encoded_df], axis=1)


def preprocess_input(df, models_dir, goal='priority'):
    # Load artifacts
    vocab    = joblib.load(f'{models_dir}/priority_vocab.pkl')
    ohe      = joblib.load(f'{models_dir}/ohe_priority.pkl' if goal == 'priority'
                           else f'{models_dir}/ohe_sla_satisfaction.pkl')
    scaler   = joblib.load(f'{models_dir}/scaler_{goal}.pkl')

    # Text
    seq = text_to_sequence(df['issue_description'].values[0], vocab)
    padded  = torch.LongTensor([pad_sequence(seq, MAX_LEN)])

    # Structured
    struct = df.drop(columns=['issue_description'])
    if goal == 'priority':
        struct = struct.drop(columns=[c for c in LEAKY_COLS_PRIORITY if c in struct.columns])
        num_cols = NUM_COLS_PRIORITY
    else:
        num_cols = NUM_COLS_SLA_SAT

    cat_cols = [c for c in struct.select_dtypes('str').columns]
    struct   = encode_cats(struct, ohe, cat_cols)
    struct[num_cols] = scaler.transform(struct[num_cols])

    struct_tensor = torch.FloatTensor(struct.values)
    return padded, struct_tensor


# ── Load models ────────────────────────────────────────────────────────────────

def load_models(models_dir):
    vocab       = joblib.load(f'{models_dir}/priority_vocab.pkl')
    vocab_size  = len(vocab)

    priority_model = PriorityModel(vocab_size, EMBED_DIM, HIDDEN_DIM, 73, 4)
    priority_model.load_state_dict(torch.load(f'{models_dir}/priority_classifier_dl.pth',
                                              map_location='cpu'))
    priority_model.eval()

    sla_model = SLAModel(vocab_size, EMBED_DIM, HIDDEN_DIM, 76, 2)
    sla_model.load_state_dict(torch.load(f'{models_dir}/sla_classifier_dl.pth',
                                         map_location='cpu'))
    sla_model.eval()

    satisfaction_model = SatisfactionModel(vocab_size, EMBED_DIM, HIDDEN_DIM, 76, 5)
    satisfaction_model.load_state_dict(torch.load(f'{models_dir}/satisfaction_classifier_dl.pth',
                                                   map_location='cpu'))
    satisfaction_model.eval()

    return priority_model, sla_model, satisfaction_model


# ── Main predict function ──────────────────────────────────────────────────────

def predict(ticket: dict) -> dict:
    config     = load_config()
    models_dir = str(ROOT / config['paths']['models_dir'])

    # Convert to DataFrame
    df = pd.DataFrame([ticket])

    # Clean and engineer
    df = clean_data(df)
    df = engineer_features(df)

    # Drop target columns if present
    df = df.drop(columns=[c for c in ['priority', 'sla_breached',
                                       'customer_satisfaction_score'] if c in df.columns])

    # Load models
    priority_model, sla_model, satisfaction_model = load_models(models_dir)

    # Priority prediction
    text_p, struct_p = preprocess_input(df.copy(), models_dir, goal='priority')
    with torch.no_grad():
        priority_pred = priority_model(text_p, struct_p).argmax(dim=1).item()

    # SLA prediction
    text_s, struct_s = preprocess_input(df.copy(), models_dir, goal='sla')
    with torch.no_grad():
        sla_pred = sla_model(text_s, struct_s).argmax(dim=1).item()

    # Satisfaction prediction
    text_sat, struct_sat = preprocess_input(df.copy(), models_dir, goal='satisfaction')
    with torch.no_grad():
        satisfaction_pred = satisfaction_model(text_sat, struct_sat).argmax(dim=1).item() + 1

    # Decode priority
    le_priority = joblib.load(f'{models_dir}/le_priority.pkl')
    priority_label = le_priority.inverse_transform([priority_pred])[0]

    return {
        'priority':     priority_label,
        'sla_breach':   'Yes' if sla_pred == 1 else 'No',
        'satisfaction': satisfaction_pred
    }


if __name__ == '__main__':
    # Test with a sample ticket
    sample_ticket = {
        'product': 'billing system',
        'category': 'payment problem',
        'issue_description': 'i am unable to process my payment and account is locked',
        'channel': 'email',
        'region': 'north america',
        'customer_age': 35,
        'customer_gender': 'male',
        'subscription_type': 'premium',
        'customer_tenure_months': 24,
        'previous_tickets': 5,
        'first_response_time_hours': 2.5,
        'escalated': 'no',
        'operating_system': 'windows',
        'browser': 'chrome',
        'payment_method': 'credit card',
        'language': 'english',
        'preferred_contact_time': 'morning',
        'customer_segment': 'individual',
        'ticket_created_date': '2024-01-15',
        'ticket_resolved_date': '2024-01-17',
        'status': 'open',
        'sla_breached': 'no',
        'customer_satisfaction_score': 3,
        'resolution_notes': '',
        'issue_complexity_score': 3,
        'resolution_time_hours': 48.0
    }

    result = predict(sample_ticket)
    print(result)
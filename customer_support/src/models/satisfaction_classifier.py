import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import load_npz
from xgboost import XGBClassifier


# ── Constants ──────────────────────────────────────────────────────────────────

MAX_LEN    = 13
EMBED_DIM  = 32
HIDDEN_DIM = 64
NUM_CLASSES = 5
BATCH_SIZE = 64
NUM_EPOCHS = 5
NUM_COLS   = ['customer_age', 'customer_tenure_months', 'previous_tickets',
              'first_response_time_hours', 'day_of_week', 'month', 'ticket_rate']


# ── Model ──────────────────────────────────────────────────────────────────────

class DualInputModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, struct_input_dim, num_classes):
        super(DualInputModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.struct_fc = nn.Sequential(
            nn.Linear(struct_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, text, struct):
        embedded       = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        text_out       = hidden[-1]
        struct_out     = self.struct_fc(struct)
        combined       = torch.cat([text_out, struct_out], dim=1)
        return self.classifier(combined)


# ── Helpers ────────────────────────────────────────────────────────────────────

def text_to_sequence(text, vocab):
    return [vocab.get(word, 0) for word in str(text).split()]


def pad_sequences(sequences, max_len):
    return np.array([seq + [0] * (max_len - len(seq)) for seq in sequences])


def encode_cats(X, ohe, cat_cols):
    encoded    = ohe.transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols), index=X.index)
    return pd.concat([X.drop(columns=cat_cols), encoded_df], axis=1)


# ── Baseline ───────────────────────────────────────────────────────────────────

def train_baseline(features_dir, models_dir):
    X_train = load_npz(f'{features_dir}/X_satisfaction_train.npz')
    X_val   = load_npz(f'{features_dir}/X_satisfaction_val.npz')
    X_test  = load_npz(f'{features_dir}/X_satisfaction_test.npz')
    y_train = np.load(f'{features_dir}/y_satisfaction_train.npy') - 1
    y_val   = np.load(f'{features_dir}/y_satisfaction_val.npy') - 1
    y_test  = np.load(f'{features_dir}/y_satisfaction_test.npy') - 1

    model = XGBClassifier(
        objective='multi:softmax',
        num_class=NUM_CLASSES,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print("Baseline Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))

    joblib.dump(model, f'{models_dir}/satisfaction_classifier_baseline.pkl')
    return model


# ── Deep Learning ──────────────────────────────────────────────────────────────

def prepare_dl_data(featured_data_path, models_dir):
    df = pd.read_csv(featured_data_path)

    text_data       = df['issue_description']
    structured_data = df.drop(columns=['issue_description', 'priority',
                                        'sla_breached', 'customer_satisfaction_score'])

    # Text sequences using saved vocab
    vocab     = joblib.load(f'{models_dir}/priority_vocab.pkl')
    sequences = [text_to_sequence(text, vocab) for text in text_data]
    padded    = pad_sequences(sequences, MAX_LEN)

    # Load saved targets
    y_train = np.load(f'{models_dir}/../data/features/y_satisfaction_train.npy')
    y_val   = np.load(f'{models_dir}/../data/features/y_satisfaction_val.npy')
    y_test  = np.load(f'{models_dir}/../data/features/y_satisfaction_test.npy')

    # Split to match saved targets
    X_text_tv, X_text_test, X_struct_tv, X_struct_test = train_test_split(
        padded, structured_data, test_size=0.15, random_state=42
    )
    X_text_train, X_text_val, X_struct_train, X_struct_val = train_test_split(
        X_text_tv, X_struct_tv, test_size=0.176, random_state=42
    )

    # Load artifacts
    ohe_satisfaction    = joblib.load(f'{models_dir}/ohe_sla_satisfaction.pkl')
    scaler_satisfaction = joblib.load(f'{models_dir}/scaler_satisfaction.pkl')

    # Encode and scale
    cat_cols = [c for c in X_struct_train.select_dtypes('str').columns]
    X_struct_train = encode_cats(X_struct_train, ohe_satisfaction, cat_cols)
    X_struct_val   = encode_cats(X_struct_val,   ohe_satisfaction, cat_cols)
    X_struct_test  = encode_cats(X_struct_test,  ohe_satisfaction, cat_cols)

    X_struct_train[NUM_COLS] = scaler_satisfaction.transform(X_struct_train[NUM_COLS])
    X_struct_val[NUM_COLS]   = scaler_satisfaction.transform(X_struct_val[NUM_COLS])
    X_struct_test[NUM_COLS]  = scaler_satisfaction.transform(X_struct_test[NUM_COLS])

    # Convert to tensors (subtract 1 for 0-indexed classes)
    return (
        vocab,
        X_struct_train.shape[1],
        torch.LongTensor(X_text_train),
        torch.LongTensor(X_text_val),
        torch.LongTensor(X_text_test),
        torch.FloatTensor(X_struct_train.values),
        torch.FloatTensor(X_struct_val.values),
        torch.FloatTensor(X_struct_test.values),
        torch.LongTensor(y_train - 1),
        torch.LongTensor(y_val - 1),
        torch.LongTensor(y_test - 1)
    )


def train_dl(featured_data_path, models_dir):
    (vocab, struct_dim,
     X_text_train, X_text_val, X_text_test,
     X_struct_train, X_struct_val, X_struct_test,
     y_train, y_val, y_test) = prepare_dl_data(featured_data_path, models_dir)

    train_loader = DataLoader(TensorDataset(X_text_train, X_struct_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_text_val,   X_struct_val,   y_val),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_text_test,  X_struct_test,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = DualInputModel(len(vocab), EMBED_DIM, HIDDEN_DIM, struct_dim, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for text_b, struct_b, y_b in train_loader:
            text_b, struct_b, y_b = text_b.to(device), struct_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out  = model(text_b, struct_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = out.max(1)
            correct += pred.eq(y_b).sum().item()
            total   += y_b.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for text_b, struct_b, y_b in val_loader:
                text_b, struct_b, y_b = text_b.to(device), struct_b.to(device), y_b.to(device)
                _, pred = model(text_b, struct_b).max(1)
                val_correct += pred.eq(y_b).sum().item()
                val_total   += y_b.size(0)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {correct/total:.4f} | "
              f"Val Acc: {val_correct/val_total:.4f}")

    # Test evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text_b, struct_b, y_b in test_loader:
            text_b, struct_b, y_b = text_b.to(device), struct_b.to(device), y_b.to(device)
            _, pred = model(text_b, struct_b).max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())

    print(f"Test Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(classification_report(all_labels, all_preds))

    torch.save(model.state_dict(), f'{models_dir}/satisfaction_classifier_dl.pth')
    return model
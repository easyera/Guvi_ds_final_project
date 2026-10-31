import sys
import yaml
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / 'src'))

from data.data_loader_saver import load_csv, save_csv
from data.data_cleaning import clean_data
from features.feature_engineering import engineer_features
from preprocessing.preprocessor import run_preprocessing
from models.priority_classifier import train_baseline as priority_baseline, train_dl as priority_dl
from models.sla_classifier import train_baseline as sla_baseline, train_dl as sla_dl
from models.satisfaction_classifier import train_baseline as satisfaction_baseline, train_dl as satisfaction_dl


def load_config():
    config_path = ROOT / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    print("=" * 50)
    print("Starting Training Pipeline")
    print("=" * 50)

    # Load config
    config = load_config()
    paths  = config['paths']

    raw_data_path      = ROOT / paths['raw_data']
    processed_data_path = ROOT / paths['processed_data']
    featured_data_path  = ROOT / paths['featured_data']
    features_dir        = str(ROOT / paths['features_dir'])
    models_dir          = str(ROOT / paths['models_dir'])

    # Step 1 — Load raw data
    print("\n[1/5] Loading raw data...")
    df = load_csv(str(raw_data_path))
    print(f"     Loaded: {df.shape}")

    # Step 2 — Clean data
    print("\n[2/5] Cleaning data...")
    df = clean_data(df)
    save_csv(df, str(processed_data_path))
    print(f"     Cleaned and saved: {df.shape}")

    # Step 3 — Feature engineering
    print("\n[3/5] Engineering features...")
    df = engineer_features(df)
    save_csv(df, str(featured_data_path))
    print(f"     Features saved: {df.shape}")

    # Step 4 — Preprocessing
    print("\n[4/5] Preprocessing...")
    run_preprocessing(str(featured_data_path), features_dir, models_dir)

    # Step 5 — Train models
    print("\n[5/5] Training models...")

    print("\n--- Priority Classification ---")
    print("  Baseline:")
    priority_baseline(features_dir, models_dir)
    print("  Deep Learning:")
    priority_dl(str(featured_data_path), models_dir)

    print("\n--- SLA Breach Prediction ---")
    print("  Baseline:")
    sla_baseline(features_dir, models_dir)
    print("  Deep Learning:")
    sla_dl(str(featured_data_path), models_dir)

    print("\n--- Customer Satisfaction Prediction ---")
    print("  Baseline:")
    satisfaction_baseline(features_dir, models_dir)
    print("  Deep Learning:")
    satisfaction_dl(str(featured_data_path), models_dir)

    print("\n" + "=" * 50)
    print("Pipeline Complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()
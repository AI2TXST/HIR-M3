import os
import sys
import time
import logging
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add modeling directory to path
sys.path.append(os.path.dirname(__file__))

from models import (
    train_lgb, train_xgb, train_cb, train_rf, train_gbdt, train_logreg,
    LGBM_AVAILABLE, XGB_AVAILABLE, CATBOOST_AVAILABLE,
    LGBM_IMPORT_ERROR, XGB_IMPORT_ERROR, CATBOOST_IMPORT_ERROR
)
from metrics import calculate_metrics, calculate_bootstrap_ci, find_optimal_threshold
from optimize_ensemble import optimize_ensemble_weights, evaluate_ensemble

for h in logging.root.handlers[:]: 
    logging.root.removeHandler(h)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TARGET_COL = "ever_readmitted"

COHORTS_FILES = {
    "Nationwide Cohort": [
        "../data/processed_final_mergedDF_condensed.csv",
        "data/processed_final_mergedDF_condensed.csv",
        "../data/processed_final_mergedDF.csv",
        "data/processed_final_mergedDF.csv"
    ],
    "Texas Cohort": [
        "../data/processed_final_mergedDF_TX_condensed.csv",
        "data/processed_final_mergedDF_TX_condensed.csv",
        "../data/processed_final_mergedDF_TX.csv",
        "data/processed_final_mergedDF_TX.csv"
    ]
}

def load_and_prep_urban_rural_data(filepath, area_type="urban", target_col=TARGET_COL):
    """
    Loads dataset and filters rows by Urban vs. Rural setting using POPPCT_URB / POPPCT_RUR.
      - area_type='urban': POPPCT_URB >= 50.0 (or POPPCT_RUR < 50.0)
      - area_type='rural': POPPCT_RUR >= 50.0 (or POPPCT_URB < 50.0)
    Applies downcasting, 50% stratified sampling, 80/20 train/test split, and scaling.
    """
    if not os.path.exists(filepath):
        logging.error(f"File not found for path: {filepath}")
        return None

    parquet_file = filepath.rsplit('.', 1)[0] + '.parquet'
    loaded = False
    if os.path.exists(parquet_file):
        try:
            logging.info(f"Loading dataset directly from Parquet: {parquet_file}...")
            df = pd.read_parquet(parquet_file)
            loaded = True
        except Exception as e:
            logging.info(f"Parquet engine unavailable ({e}). Falling back to CSV.")

    if not loaded:
        logging.info(f"Loading dataset directly from CSV: {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)

    # 1. Urban vs Rural Slicing
    urb_col = next((c for c in ['POPPCT_URB', 'POPPCT_URB_condensed', 'urban_pct'] if c in df.columns), None)
    rur_col = next((c for c in ['POPPCT_RUR', 'POPPCT_RUR_condensed', 'rural_pct'] if c in df.columns), None)

    initial_count = len(df)

    if area_type.lower() == "urban":
        if urb_col:
            df = df[df[urb_col] >= 50.0].copy()
        elif rur_col:
            df = df[df[rur_col] < 50.0].copy()
        else:
            logging.warning("No POPPCT_URB or POPPCT_RUR column found for Urban slicing. Using full dataset.")
    elif area_type.lower() == "rural":
        if rur_col:
            df = df[df[rur_col] >= 50.0].copy()
        elif urb_col:
            df = df[df[urb_col] < 50.0].copy()
        else:
            logging.warning("No POPPCT_URB or POPPCT_RUR column found for Rural slicing. Using full dataset.")

    sliced_count = len(df)
    logging.info(f"Filtered dataset for area_type='{area_type.upper()}': {sliced_count:,} / {initial_count:,} rows")

    if sliced_count < 100:
        logging.error(f"Insufficient rows for area_type='{area_type}' ({sliced_count} rows). Skipping.")
        return None

    # Downcast int64 to int8/int16 to save memory
    int_cols = df.select_dtypes(include=['int64', 'int32']).columns
    for c in int_cols:
        c_min, c_max = df[c].min(), df[c].max()
        if c_min >= -128 and c_max <= 127:
            df[c] = df[c].astype(np.int8)
        elif c_min >= -32768 and c_max <= 32767:
            df[c] = df[c].astype(np.int16)
        else:
            df[c] = df[c].astype(np.int32)
    
    # Downcast float64 to float32
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)

    gc.collect()

    possible_targets = [target_col, 'ever_readmitted', 'READMISSION', 'Readmission']
    t_col = next((t for t in possible_targets if t in df.columns), None)

    if not t_col:
        logging.error(f"No target column found in dataset")
        return None

    # Apply 20% stratified sampling
    logging.info(f"Sampling {area_type.upper()} dataset from {len(df):,} to 20% ({int(len(df) * 0.20):,} rows)...")
    if t_col in df.columns:
        df, _ = train_test_split(df, train_size=0.20, stratify=df[t_col], random_state=42)
    else:
        df = df.sample(frac=0.20, random_state=42).copy()

    total_sample_size = len(df)
    logging.info(f"Sampled {area_type.upper()} dataset shape: {df.shape}")

    drop_cols = ['BENE_ID', 'Beneficiary_ID', 'Assessment_Effective_Date', 'COUNTYFIPS', t_col]
    feature_cols = [c for c in df.columns if c not in drop_cols and not c.lower().endswith('id')]

    X = df[feature_cols].astype(np.float32)
    y = df[t_col].values.astype(np.int8)

    del df
    gc.collect()

    # Train / Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    del X
    gc.collect()

    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train).astype(np.float32), columns=feature_cols)
    del X_train
    gc.collect()

    X_test_scaled = pd.DataFrame(scaler.transform(X_test).astype(np.float32), columns=feature_cols)
    del X_test
    gc.collect()

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, total_sample_size

def run_urban_rural_pipeline(selected_cohort="all", output_dir="results"):
    """
    Runs model training & evaluation for each cohort split into Urban vs. Rural sub-cohorts.
    """
    logging.info("=================================================================")
    logging.info("  STARTING URBAN VS. RURAL MODELING PIPELINE")
    logging.info("=================================================================")
    os.makedirs(output_dir, exist_ok=True)

    all_model_results = []
    all_bootstrap_results = []

    # Find valid files for cohorts
    cohort_target_files = {}
    for c_name, paths in COHORTS_FILES.items():
        found = next((p for p in paths if os.path.exists(p)), None)
        if found:
            cohort_target_files[c_name] = found

    if not cohort_target_files:
        logging.error("No valid dataset files found for Urban vs. Rural modeling!")
        return

    # Filter cohorts if user specified
    if selected_cohort.lower().startswith("tx") or selected_cohort.lower().startswith("texas"):
        cohort_target_files = {k: v for k, v in cohort_target_files.items() if "Texas" in k}
    elif selected_cohort.lower().startswith("nat") or selected_cohort.lower().startswith("nationwide"):
        cohort_target_files = {k: v for k, v in cohort_target_files.items() if "Nationwide" in k}

    # Run for both Urban and Rural splits per cohort
    for base_cohort_name, filepath in cohort_target_files.items():
        for area_type in ["urban", "rural"]:
            sub_cohort_name = f"{base_cohort_name} - {area_type.capitalize()}"
            logging.info(f"\n=========================================================")
            logging.info(f"   PROCESSING SUB-COHORT: {sub_cohort_name}")
            logging.info(f"=========================================================")

            data = load_and_prep_urban_rural_data(filepath, area_type=area_type)
            if data is None:
                continue

            X_train, X_test, y_train, y_test, feature_cols, cohort_sample_size = data
            w_train = np.ones_like(y_train, dtype=float)
            w_test = np.ones_like(y_test, dtype=float)

            y_prob_dict = {}

            # 1. LightGBM
            if LGBM_AVAILABLE:
                logging.info("Training LightGBM...")
                try:
                    prob = train_lgb(X_train, y_train, w_train, X_test, y_test, {})
                    if prob is not None: y_prob_dict['LightGBM'] = np.asarray(prob).ravel()
                except Exception as e:
                    logging.error(f"LightGBM Error: {e}")

            # 2. XGBoost
            if XGB_AVAILABLE:
                logging.info("Training XGBoost...")
                try:
                    prob = train_xgb(X_train, y_train, w_train, X_test, y_test, {})
                    if prob is not None: y_prob_dict['XGBoost'] = np.asarray(prob).ravel()
                except Exception as e:
                    logging.error(f"XGBoost Error: {e}")

            # 3. CatBoost
            if CATBOOST_AVAILABLE:
                logging.info("Training CatBoost...")
                try:
                    prob = train_cb(X_train, y_train, w_train, X_test, y_test, {})
                    if prob is not None: y_prob_dict['CatBoost'] = np.asarray(prob).ravel()
                except Exception as e:
                    logging.error(f"CatBoost Error: {e}")

            # 4. Random Forest
            logging.info("Training Random Forest...")
            try:
                prob = train_rf(X_train, y_train, w_train, X_test, y_test, {})
                if prob is not None: y_prob_dict['Random Forest'] = np.asarray(prob).ravel()
            except Exception as e:
                logging.error(f"Random Forest Error: {e}")

            # 5. Gradient Boosting (GBT)
            logging.info("Training Gradient Boosting (GBT)...")
            try:
                prob = train_gbdt(X_train, y_train, w_train, X_test, y_test, {})
                if prob is not None: y_prob_dict['Gradient Boosting'] = np.asarray(prob).ravel()
            except Exception as e:
                logging.error(f"Gradient Boosting Error: {e}")

            # 6. Logistic Regression
            logging.info("Training Logistic Regression...")
            try:
                prob = train_logreg(X_train, y_train, w_train, X_test, y_test, {})
                if prob is not None: y_prob_dict['Logistic Regression'] = np.asarray(prob).ravel()
            except Exception as e:
                logging.error(f"Logistic Regression Error: {e}")

            # Evaluate baseline ML models for this sub-cohort

            # Evaluate models for this sub-cohort
            model_results_cohort = []
            boot_results_cohort = []

            for model_name, y_prob in y_prob_dict.items():
                thresh, _ = find_optimal_threshold(y_test, y_prob, metric='f1')
                m = calculate_metrics(y_test, y_prob, threshold=thresh)
                m['Cohort'] = sub_cohort_name
                m['Setting'] = area_type.capitalize()
                m['Cohort Size'] = cohort_sample_size
                m['Num Features'] = len(feature_cols)
                m['Model'] = model_name
                all_model_results.append(m)
                model_results_cohort.append(m)

                # Bootstrap 95% CIs
                boot_summary = calculate_bootstrap_ci(y_test, y_prob, threshold=thresh, n_bootstrap=200)
                for m_name, b_data in boot_summary.items():
                    b_item = {
                        'Cohort': sub_cohort_name,
                        'Setting': area_type.capitalize(),
                        'Cohort Size': cohort_sample_size,
                        'Num Features': len(feature_cols),
                        'Model': model_name,
                        'Metric': m_name,
                        'Mean': b_data['Mean'],
                        'CI_Lower': b_data['CI_Lower'],
                        'CI_Upper': b_data['CI_Upper'],
                        'Formatted': b_data['Formatted']
                    }
                    all_bootstrap_results.append(b_item)
                    boot_results_cohort.append(b_item)

            # Weighted Ensemble Optimization
            if len(y_prob_dict) >= 2:
                logging.info(f"Optimizing Weighted Ensemble for {sub_cohort_name}...")
                try:
                    best_weights, opt_thresh, _ = optimize_ensemble_weights(y_test, y_prob_dict)
                    ensemble_prob, ens_metrics = evaluate_ensemble(y_test, y_prob_dict, weights=best_weights, threshold=opt_thresh)
                    ens_metrics['Cohort'] = sub_cohort_name
                    ens_metrics['Setting'] = area_type.capitalize()
                    ens_metrics['Cohort Size'] = cohort_sample_size
                    ens_metrics['Num Features'] = len(feature_cols)
                    ens_metrics['Model'] = 'Weighted Ensemble'
                    all_model_results.append(ens_metrics)
                    model_results_cohort.append(ens_metrics)

                    boot_summary = calculate_bootstrap_ci(y_test, ensemble_prob, threshold=opt_thresh, n_bootstrap=200)
                    for m_name, b_data in boot_summary.items():
                        b_item = {
                            'Cohort': sub_cohort_name,
                            'Setting': area_type.capitalize(),
                            'Cohort Size': cohort_sample_size,
                            'Num Features': len(feature_cols),
                            'Model': 'Weighted Ensemble',
                            'Metric': m_name,
                            'Mean': b_data['Mean'],
                            'CI_Lower': b_data['CI_Lower'],
                            'CI_Upper': b_data['CI_Upper'],
                            'Formatted': b_data['Formatted']
                        }
                        all_bootstrap_results.append(b_item)
                        boot_results_cohort.append(b_item)
                except Exception as e:
                    logging.error(f"Ensemble Optimization Error: {e}")

            # Save cohort specific output file
            prefix = sub_cohort_name.replace(" ", "_").replace("-", "_")
            if model_results_cohort:
                c_df = pd.DataFrame(model_results_cohort)
                c_path = os.path.join(output_dir, f"{prefix}_modeling_results.csv")
                c_df.to_csv(c_path, index=False)
                logging.info(f"Saved {sub_cohort_name} modeling results to {c_path}")

            del X_train, X_test, y_train, y_test
            gc.collect()

    # Save consolidated results across all urban/rural sub-cohorts
    if all_model_results:
        results_df = pd.DataFrame(all_model_results)
        results_path = os.path.join(output_dir, "urban_rural_modeling_results.csv")
        results_df.to_csv(results_path, index=False)
        logging.info(f"\nSaved consolidated Urban vs. Rural modeling results to {results_path}")

    if all_bootstrap_results:
        boot_df = pd.DataFrame(all_bootstrap_results)
        boot_path = os.path.join(output_dir, "urban_rural_bootstrap_results.csv")
        boot_df.to_csv(boot_path, index=False)
        logging.info(f"Saved consolidated Urban vs. Rural bootstrap results to {boot_path}")

    logging.info("=== Urban vs. Rural Modeling Pipeline Finished Successfully ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Urban vs. Rural Sub-Cohort Modeling Pipeline")
    parser.add_argument("--cohort", choices=["Texas", "Nationwide", "all"], default="all",
                        help="Specify cohort to train (Texas, Nationwide, or all)")
    args = parser.parse_args()

    run_urban_rural_pipeline(args.cohort)

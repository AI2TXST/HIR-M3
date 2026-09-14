import os
import sys
import time
import logging
import gc
import json
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
        "../data/processed_final_mergedDF_condensed.parquet",
        "data/processed_final_mergedDF_condensed.parquet",
        "../data/processed_final_mergedDF_condensed.csv",
        "data/processed_final_mergedDF_condensed.csv"
    ],
    "Texas Cohort": [
        "../data/processed_final_mergedDF_condensed_TX.parquet",
        "data/processed_final_mergedDF_condensed_TX.parquet",
        "../data/processed_final_mergedDF_condensed_TX.csv",
        "data/processed_final_mergedDF_condensed_TX.csv"
    ]
}

def load_and_prep_urban_rural_data(filepath, area_type="urban", target_col=TARGET_COL):
    """
    Loads dataset (preferring Parquet or CSV chunks), filters rows by Urban vs. Rural setting,
    applies downcasting, 20% stratified sampling for nationwide cohort, 80/20 train/test split, and scaling.
    """
    actual_path = None
    if isinstance(filepath, (list, tuple)):
        actual_path = next((p for p in filepath if os.path.exists(p)), None)
    elif os.path.exists(str(filepath)):
        actual_path = str(filepath)

    if not actual_path:
        logging.error(f"File not found for path: {filepath}")
        return None

    filepath = actual_path

    if filepath.endswith('.parquet'):
        logging.info(f"Loading dataset from Parquet: {filepath}...")
        df = pd.read_parquet(filepath)
        initial_count = len(df)
        urb_col = next((c for c in ['POPPCT_URB', 'POPPCT_URB_condensed', 'urban_pct'] if c in df.columns), None)
        rur_col = next((c for c in ['POPPCT_RUR', 'POPPCT_RUR_condensed', 'rural_pct'] if c in df.columns), None)
        ruca_c = next((c for c in ['RUCA_Category', 'ruca_category', 'ruca'] if c in df.columns), None)

        if area_type.lower() == "urban":
            if urb_col:
                df = df[df[urb_col] >= 50.0]
            elif rur_col:
                df = df[df[rur_col] < 50.0]
            elif ruca_c:
                df = df[df[ruca_c].astype(str).str.lower().isin(['urban', '1', '1.0'])]
        elif area_type.lower() == "rural":
            if rur_col:
                df = df[df[rur_col] >= 50.0]
            elif urb_col:
                df = df[df[urb_col] < 50.0]
            elif ruca_c:
                df = df[df[ruca_c].astype(str).str.lower().isin(['rural', '2', '2.0', '3', '3.0'])]

        if len(df) == 0:
            logging.error(f"No rows matched filter for area_type='{area_type}' in {filepath}")
            return None
    else:
        logging.info(f"Loading dataset in memory-efficient chunks from CSV: {filepath}...")
        sample_df = pd.read_csv(filepath, nrows=5)
        urb_col = next((c for c in ['POPPCT_URB', 'POPPCT_URB_condensed', 'urban_pct'] if c in sample_df.columns), None)
        rur_col = next((c for c in ['POPPCT_RUR', 'POPPCT_RUR_condensed', 'rural_pct'] if c in sample_df.columns), None)
        ruca_c = next((c for c in ['RUCA_Category', 'ruca_category', 'ruca'] if c in sample_df.columns), None)

        chunks = []
        initial_count = 0

        for chunk in pd.read_csv(filepath, chunksize=100000, low_memory=False):
            initial_count += len(chunk)

            if area_type.lower() == "urban":
                if urb_col:
                    chunk = chunk[chunk[urb_col] >= 50.0]
                elif rur_col:
                    chunk = chunk[chunk[rur_col] < 50.0]
                elif ruca_c:
                    chunk = chunk[chunk[ruca_c].astype(str).str.lower().isin(['urban', '1', '1.0'])]
            elif area_type.lower() == "rural":
                if rur_col:
                    chunk = chunk[chunk[rur_col] >= 50.0]
                elif urb_col:
                    chunk = chunk[chunk[urb_col] < 50.0]
                elif ruca_c:
                    chunk = chunk[chunk[ruca_c].astype(str).str.lower().isin(['rural', '2', '2.0', '3', '3.0'])]

            if len(chunk) == 0:
                continue

            int_cols = chunk.select_dtypes(include=['int64', 'int32']).columns
            for c in int_cols:
                c_min, c_max = chunk[c].min(), chunk[c].max()
                if c_min >= -128 and c_max <= 127:
                    chunk[c] = chunk[c].astype(np.int8)
                elif c_min >= -32768 and c_max <= 32767:
                    chunk[c] = chunk[c].astype(np.int16)
                else:
                    chunk[c] = chunk[c].astype(np.int32)
            
            float_cols = chunk.select_dtypes(include=['float64']).columns
            if len(float_cols) > 0:
                chunk[float_cols] = chunk[float_cols].astype(np.float32)

            chunks.append(chunk)

        if not chunks:
            logging.error(f"No rows matched filter for area_type='{area_type}' in {filepath}")
            return None

        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()

    sliced_count = len(df)
    logging.info(f"Filtered dataset for area_type='{area_type.upper()}': {sliced_count:,} / {initial_count:,} rows")

    if sliced_count < 100:
        logging.error(f"Insufficient rows for area_type='{area_type}' ({sliced_count} rows). Skipping.")
        return None

    possible_targets = [target_col, 'ever_readmitted', 'READMISSION', 'Readmission']
    t_col = next((t for t in possible_targets if t in df.columns), None)

    if not t_col:
        logging.error(f"No target column found in dataset")
        return None

    # Perform 50% stratified sampling only for Nationwide dataset; keep Texas at 100%
    if "TX" not in str(filepath) and "texas" not in str(filepath).lower():
        logging.info(f"Sampling {area_type.upper()} Nationwide dataset from {len(df):,} to 50% ({int(len(df) * 0.50):,} rows)...")
        if t_col in df.columns:
            df, _ = train_test_split(df, train_size=0.50, stratify=df[t_col], random_state=42)
        else:
            df = df.sample(frac=0.50, random_state=42).copy()
    else:
        logging.info(f"Using 100% of Texas dataset for {area_type.upper()} ({len(df):,} rows)...")

    total_sample_size = len(df)
    logging.info(f"Final {area_type.upper()} dataset shape for modeling: {df.shape}")

    drop_cols = [
        'BENE_ID', 'Beneficiary_ID', 'Patient_ID', 'Assessment_Effective_Date', 'COUNTYFIPS',
        'Days_Cared_For', 'ever_deceased', 'NumVisits', 'DaysBetweenVisits',
        'PrevVisitDate', 'Last_Assessment_Date', t_col
    ]
    id_cols = [c for c in df.columns if 'beneficiary' in c.lower() or 'bene_id' in c.lower() or c.lower().endswith('id')]
    drop_cols = list(set(drop_cols).union(set(id_cols)))
    feature_cols = [c for c in df.columns if c not in drop_cols]

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
        # Load saved tuned ML hyperparameters if available
        prefix = base_cohort_name.lower().replace(" ", "_")
        candidate_hp_paths = [
            os.path.join(output_dir, f"{prefix}_best_hyperparameters.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"{prefix}_best_hyperparameters.json"),
            os.path.join("results", f"{prefix}_best_hyperparameters.json"),
            os.path.join("modeling", "results", f"{prefix}_best_hyperparameters.json")
        ]
        ml_hp_path = next((p for p in candidate_hp_paths if os.path.exists(p)), None)
        ml_hp = {}
        if ml_hp_path:
            try:
                with open(ml_hp_path) as f:
                    ml_hp = json.load(f)
                logging.info(f"Loaded tuned ML hyperparameters for {base_cohort_name} from {ml_hp_path}")
            except Exception as e:
                logging.warning(f"Error loading {ml_hp_path}: {e}")
        else:
            logging.warning(f"Could not find tuned hyperparameter file for {base_cohort_name} in candidate paths: {candidate_hp_paths}")

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
                cfg = ml_hp.get('LightGBM', {})
                logging.info(f"Training LightGBM (Params: {cfg})...")
                try:
                    prob = train_lgb(X_train, y_train, w_train, X_test, y_test, cfg)
                    if prob is not None: y_prob_dict['LightGBM'] = np.asarray(prob).ravel()
                except Exception as e:
                    logging.error(f"LightGBM Error: {e}")

            # 2. XGBoost
            if XGB_AVAILABLE:
                cfg = ml_hp.get('XGBoost', {})
                logging.info(f"Training XGBoost (Params: {cfg})...")
                try:
                    prob = train_xgb(X_train, y_train, w_train, X_test, y_test, cfg)
                    if prob is not None: y_prob_dict['XGBoost'] = np.asarray(prob).ravel()
                except Exception as e:
                    logging.error(f"XGBoost Error: {e}")

            # 3. CatBoost
            if CATBOOST_AVAILABLE:
                cfg = ml_hp.get('CatBoost', {})
                logging.info(f"Training CatBoost (Params: {cfg})...")
                try:
                    prob = train_cb(X_train, y_train, w_train, X_test, y_test, cfg)
                    if prob is not None: y_prob_dict['CatBoost'] = np.asarray(prob).ravel()
                except Exception as e:
                    logging.error(f"CatBoost Error: {e}")

            # 4. Random Forest
            cfg = ml_hp.get('Random Forest', {})
            logging.info(f"Training Random Forest (Params: {cfg})...")
            try:
                prob = train_rf(X_train, y_train, w_train, X_test, y_test, cfg)
                if prob is not None: y_prob_dict['Random Forest'] = np.asarray(prob).ravel()
            except Exception as e:
                logging.error(f"Random Forest Error: {e}")

            # 5. Gradient Boosting (GBT)
            cfg = ml_hp.get('Gradient Boosting', {})
            logging.info(f"Training Gradient Boosting (Params: {cfg})...")
            try:
                prob = train_gbdt(X_train, y_train, w_train, X_test, y_test, cfg)
                if prob is not None: y_prob_dict['Gradient Boosting'] = np.asarray(prob).ravel()
            except Exception as e:
                logging.error(f"Gradient Boosting Error: {e}")

            # 6. Logistic Regression
            cfg = ml_hp.get('Logistic Regression', {})
            logging.info(f"Training Logistic Regression (Params: {cfg})...")
            try:
                prob = train_logreg(X_train, y_train, w_train, X_test, y_test, cfg)
                if prob is not None: y_prob_dict['Logistic Regression'] = np.asarray(prob).ravel()
            except Exception as e:
                logging.error(f"Logistic Regression Error: {e}")

            # Predict validation probabilities for threshold tuning (no test set leakage)
            X_tr, X_val, y_tr, y_val, w_tr_split, w_val_split = train_test_split(
                X_train, y_train, w_train, test_size=0.2, stratify=y_train, random_state=42
            )
            y_val_prob_dict = {}
            if LGBM_AVAILABLE:
                try:
                    prob_v = train_lgb(X_tr, y_tr, w_tr_split, X_val, y_val, ml_hp.get('LightGBM', {}))
                    if prob_v is not None: y_val_prob_dict['LightGBM'] = np.asarray(prob_v).ravel()
                except Exception: pass
            if XGB_AVAILABLE:
                try:
                    prob_v = train_xgb(X_tr, y_tr, w_tr_split, X_val, y_val, ml_hp.get('XGBoost', {}))
                    if prob_v is not None: y_val_prob_dict['XGBoost'] = np.asarray(prob_v).ravel()
                except Exception: pass
            if CATBOOST_AVAILABLE:
                try:
                    prob_v = train_cb(X_tr, y_tr, w_tr_split, X_val, y_val, ml_hp.get('CatBoost', {}))
                    if prob_v is not None: y_val_prob_dict['CatBoost'] = np.asarray(prob_v).ravel()
                except Exception: pass
            try:
                prob_v = train_rf(X_tr, y_tr, w_tr_split, X_val, y_val, ml_hp.get('Random Forest', {}))
                if prob_v is not None: y_val_prob_dict['Random Forest'] = np.asarray(prob_v).ravel()
            except Exception: pass
            try:
                prob_v = train_gbdt(X_tr, y_tr, w_tr_split, X_val, y_val, ml_hp.get('Gradient Boosting', {}))
                if prob_v is not None: y_val_prob_dict['Gradient Boosting'] = np.asarray(prob_v).ravel()
            except Exception: pass
            try:
                prob_v = train_logreg(X_tr, y_tr, w_tr_split, X_val, y_val, ml_hp.get('Logistic Regression', {}))
                if prob_v is not None: y_val_prob_dict['Logistic Regression'] = np.asarray(prob_v).ravel()
            except Exception: pass

            # Evaluate baseline ML models for this sub-cohort
            model_results_cohort = []
            boot_results_cohort = []

            for model_name, y_prob in y_prob_dict.items():
                if model_name in y_val_prob_dict and y_val_prob_dict[model_name] is not None:
                    thresh, _ = find_optimal_threshold(y_val, y_val_prob_dict[model_name], metric='f1')
                else:
                    thresh = 0.5

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

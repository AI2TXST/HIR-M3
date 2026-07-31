import pandas as pd
import numpy as np
import os
import sys
import time
import logging
import warnings
from joblib import Parallel, delayed

# Sklearn Imports
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import VarianceThreshold, RFE, SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, silhouette_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

# ML Libraries that might be optional
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True

    class FeatureEmbedding(nn.Module):
        def __init__(self, num_features, embed_dim):
            super().__init__()
            self.num_features = num_features
            self.embed_dim = embed_dim
            self.weight = nn.Parameter(torch.randn(num_features, embed_dim) * 0.02)
            self.bias = nn.Parameter(torch.zeros(num_features, embed_dim))
            
        def forward(self, x):
            # Vectorized projection: (Batch, NumFeatures, 1) * (1, NumFeatures, EmbedDim) + (1, NumFeatures, EmbedDim)
            return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

    class FeatureDropout(nn.Module):
        def __init__(self, p=0.1):
            super().__init__()
            self.p = p
            
        def forward(self, x):
            if not self.training or self.p == 0.0:
                return x
            mask = torch.rand(x.size(0), x.size(1), 1, device=x.device) >= self.p
            return x * mask / (1.0 - self.p)

    class HierarchicalAttention(nn.Module):
        def __init__(self, embed_dim, num_heads, dropout=0.1):
            super().__init__()
            self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(dropout)
            )
            
        def forward(self, x, need_weights=False):
            if need_weights:
                attn_output, attn_weights = self.mha(x, x, x, need_weights=True, average_attn_weights=True)
            else:
                attn_output, attn_weights = self.mha(x, x, x, need_weights=False)
            x = self.norm1(x + attn_output)
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
            return x, attn_weights

    class GatedMLPBlock(nn.Module):
        def __init__(self, in_dim, out_dim, dropout=0.1):
            super().__init__()
            self.fc = nn.Linear(in_dim, out_dim)
            self.gate = nn.Linear(in_dim, out_dim)
            self.norm = nn.LayerNorm(out_dim)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            
        def forward(self, x):
            gate_val = torch.sigmoid(self.gate(x))
            fc_val = self.act(self.fc(x))
            out = gate_val * fc_val
            return self.drop(self.norm(out))

    class HIRModel(nn.Module):
        def __init__(self, num_features, embed_dim=32, num_heads=4, hidden_dim=64, 
                     num_layers=2, feat_drop=0.0, attn_drop=0.1, head_drop=0.1):
            super().__init__()
            self.embedding = FeatureEmbedding(num_features, embed_dim)
            self.feat_dropout = FeatureDropout(feat_drop)
            
            self.attention_layers = nn.ModuleList([
                HierarchicalAttention(embed_dim, num_heads, dropout=attn_drop)
                for _ in range(num_layers)
            ])
            
            self.head = nn.Sequential(
                GatedMLPBlock(embed_dim, hidden_dim, dropout=head_drop),
                GatedMLPBlock(hidden_dim, hidden_dim // 2, dropout=head_drop),
                nn.Linear(hidden_dim // 2, 1)
            )
            
        def forward(self, x, need_weights=False):
            embeddings = self.embedding(x)
            embeddings = self.feat_dropout(embeddings)
            attn_list = []
            x = embeddings
            for layer in self.attention_layers:
                x, attn_weights = layer(x, need_weights=need_weights)
                if need_weights and attn_weights is not None:
                    attn_list.append(attn_weights)
                
            pooled = x.mean(dim=1)
            logits = self.head(pooled)
            mean_attn = attn_list[-1] if (need_weights and attn_list) else None
            return logits, mean_attn

except ImportError:
    TORCH_AVAILABLE = False

try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
def log_time_and_step(step_name):
    timestamp = time.ctime()
    print(f"[{timestamp}] {step_name}", flush=True)
    logging.info(f"{step_name}")

# ---------------------------------------------------------
# FEATURE SELECTION METHODS
# ---------------------------------------------------------
def variance_threshold_fs(X_train, y_train, X_test, y_test):
    # Calculate variances across all features
    vt = VarianceThreshold(threshold=0.0) 
    try:
        vt.fit(X_train)
        variances = vt.variances_
        indices = np.argsort(variances)[::-1]
        return {'method': 'Variance Threshold', 'indices': indices, 'scores': variances[indices], 'score_type': 'Variance'}
    except: return None

def lasso_selection(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    X_tr_eval, y_tr_eval = X_train, y_train
    if len(X_train) > 100000:
        idx = np.random.RandomState(42).choice(len(X_train), 100000, replace=False)
        X_tr_eval = X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx]
        y_tr_eval = y_train[idx]
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.01, random_state=42)
    lasso.fit(X_tr_eval, y_tr_eval)
    scores = lasso.coef_[0]
    indices = np.argsort(np.abs(scores))[::-1]
    return {'method': 'Lasso (L1 LogReg)', 'indices': indices, 'scores': scores[indices], 'score_type': 'Coefficient'}

def rf_importance(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_importances = rf.feature_importances_
    rf_features = np.argsort(rf_importances)[::-1]
    rf_scores = rf_importances[rf_features]
    return {'method': 'Random Forest Importance', 'indices': rf_features, 'scores': rf_scores, 'score_type': 'Importance'}

def permutation_importance_rf(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    if len(X_test) > 1000:
        idx = np.random.RandomState(1).choice(len(X_test), 1000, replace=False)
        X_eval, y_eval = X_test.iloc[idx], y_test[idx]
    else:
        X_eval, y_eval = X_test, y_test
    perm_importance_rf = permutation_importance(rf, X_eval, y_eval, n_repeats=2, random_state=1, n_jobs=2)
    perm_features_rf = perm_importance_rf.importances_mean.argsort()[::-1]
    perm_scores_rf = perm_importance_rf.importances_mean[perm_features_rf]
    return {'method': 'Permutation Importance (RF)', 'indices': perm_features_rf, 'scores': perm_scores_rf, 'score_type': 'Importance'}

def permutation_importance_ridge(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    ridge = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=500)
    ridge.fit(X_train, y_train)
    if len(X_test) > 1000:
        idx = np.random.RandomState(1).choice(len(X_test), 1000, replace=False)
        X_eval, y_eval = X_test.iloc[idx], y_test[idx]
    else:
        X_eval, y_eval = X_test, y_test
    perm_importance_ridge = permutation_importance(ridge, X_eval, y_eval, n_repeats=2, random_state=1, n_jobs=2)
    perm_features_ridge = perm_importance_ridge.importances_mean.argsort()[::-1]
    perm_scores_ridge = perm_importance_ridge.importances_mean[perm_features_ridge]
    return {'method': 'Permutation Importance (Ridge/LogReg)', 'indices': perm_features_ridge, 'scores': perm_scores_ridge, 'score_type': 'Importance'}

def lgb_importance(X_train, y_train, X_test, y_test):
    if not LGBM_AVAILABLE: return None
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=1, verbose=-1, n_jobs=-1)
    lgb_model.fit(X_train, y_train)
    scores = lgb_model.feature_importances_
    indices = np.argsort(scores)[::-1]
    return {'method': 'LightGBM Importance', 'indices': indices, 'scores': scores[indices], 'score_type': 'Importance'}

def xgb_importance(X_train, y_train, X_test, y_test):
    if not XGB_AVAILABLE: return None
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=1, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    scores = xgb_model.feature_importances_
    indices = np.argsort(scores)[::-1]
    return {'method': 'XGBoost Importance', 'indices': indices, 'scores': scores[indices], 'score_type': 'Importance'}

def hir_importance(X_train, y_train, X_test, y_test, epochs=3, batch_size=256):
    if not TORCH_AVAILABLE: return None
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tr_np = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
        y_tr_np = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        X_te_np = X_test.values if hasattr(X_test, 'values') else np.array(X_test)
        
        # Subsample training to 25,000 samples & test to 2,500 for fast, memory-safe attention ranking
        if len(X_tr_np) > 25000:
            idx_tr = np.random.RandomState(42).choice(len(X_tr_np), 25000, replace=False)
            X_tr_np, y_tr_np = X_tr_np[idx_tr], y_tr_np[idx_tr]
        if len(X_te_np) > 2500:
            idx_te = np.random.RandomState(42).choice(len(X_te_np), 2500, replace=False)
            X_te_np = X_te_np[idx_te]

        num_features = X_tr_np.shape[1]
        model = HIRModel(num_features=num_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        model.train()
        n_samples = len(X_tr_np)
        for epoch in range(epochs):
            permutation = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x = torch.tensor(X_tr_np[indices], dtype=torch.float32).to(device)
                batch_y = torch.tensor(y_tr_np[indices], dtype=torch.float32).unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        all_attn = []
        with torch.no_grad():
            for i in range(0, len(X_te_np), batch_size):
                batch_x = torch.tensor(X_te_np[i:i+batch_size], dtype=torch.float32).to(device)
                _, attn = model(batch_x)
                avg_attn_batch = attn.mean(dim=0).cpu().numpy()
                all_attn.append(avg_attn_batch * len(batch_x))
                
        global_attn_matrix = np.sum(all_attn, axis=0) / len(X_te_np)
        feature_importance = global_attn_matrix.sum(axis=0)
        
        if np.max(feature_importance) > np.min(feature_importance):
            feature_importance = (feature_importance - np.min(feature_importance)) / (np.max(feature_importance) - np.min(feature_importance))
        else:
            feature_importance = np.ones_like(feature_importance)
            
        indices = np.argsort(feature_importance)[::-1]
        return {
            'method': 'HIR-M3 Attention Importance',
            'indices': indices,
            'scores': feature_importance[indices],
            'score_type': 'Attention Importance'
        }
    except Exception as e:
        print(f"    HIR Attention Error: {e}")
        return None

AVAILABLE_FS_METHODS = {
    'Variance Threshold': variance_threshold_fs,
    'Lasso (L1 LogReg)': lasso_selection,
    'Random Forest Importance': rf_importance,
    'Permutation Importance (RF)': permutation_importance_rf,
    'Permutation Importance (Ridge/LogReg)': permutation_importance_ridge,
    'LightGBM Importance': lgb_importance,
    'XGBoost Importance': xgb_importance,
    'HIR-M3 Attention Importance': hir_importance
}

# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------
def balance_data(X, y, random_state=42):
    if IMBLEARN_AVAILABLE:
        try:
            unique, counts = np.unique(y, return_counts=True)
            if len(counts) < 2: return X, y
            ros = RandomOverSampler(random_state=random_state)
            return ros.fit_resample(X, y)
        except Exception:
            pass

    # Native pandas/numpy random oversampling fallback
    np.random.seed(random_state)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    max_count = np.max(counts)
    
    X_list, y_list = [], []
    is_df = isinstance(X, pd.DataFrame)
    
    for cls in classes:
        mask = (y == cls)
        X_cls = X[mask] if is_df else X[mask]
        y_cls = y[mask]
        
        needed = max_count - len(y_cls)
        if needed > 0:
            idxs = np.random.choice(len(y_cls), size=needed, replace=True)
            if is_df:
                X_over = pd.concat([X_cls, X_cls.iloc[idxs]], axis=0)
            else:
                X_over = np.vstack([X_cls, X_cls[idxs]])
            y_over = np.concatenate([y_cls, y_cls[idxs]])
        else:
            X_over = X_cls
            y_over = y_cls
            
        X_list.append(X_over)
        y_list.append(y_over)
        
    if is_df:
        X_resampled = pd.concat(X_list, axis=0).reset_index(drop=True)
    else:
        X_resampled = np.vstack(X_list)
    y_resampled = np.concatenate(y_list)
    
    return X_resampled, y_resampled
# ---------------------------------------------------------
# M3 HKAN Helpers
# ---------------------------------------------------------
def get_m3_feature_groups(all_features):
    """
    Returns (micro, meso, macro) lists based on the provided feature names.
    """
    # Ensure all features are strings to prevent hashing errors with numpy arrays
    all_features_str = [str(f) for f in all_features]
    feats = set(all_features_str)

    # Helper to check existence to avoid KeyErrors
    def valid(names):
        return [f for f in names if f in feats]
    
    MICRO_FEATURES = valid([
        # Demographics
        'Age', 'American_Indian_or_Alaska_Native', 'Asian', 'Black_or_African_American',
        'Hispanic_or_Latino', 'Native_Hawiian_or_Pacific_Islander', 'White',

        # Clinical utilization
        'Days_Cared_For',
        'BMI_Category_Obese-Class1', 'BMI_Category_Obese-Class2', 
        'BMI_Category_Obese-Class3', 'BMI_Category_Overweight', 'BMI_Category_Underweight',
        # Discipline
        'ByDiscipline_PT', 'ByDiscipline_RN', 'ByDiscipline_SLP/ST',

        # Comorbidities (Charlson / Elixhauser-style)
        'charlson_score', 'charlson_ageadj', 'elix_quan_score', 'aids', 'alcohol', 'ami', 
        'canc', 'carit', 'cevd','chf','coag','copd','dane','dementia','depre','drug', 'fed',
        'hp','hypc','hypothy','hypunc','ld','lymph','metacanc','obes','ond','pcd','psycho',
        'pvd', 'rend','rheumd','valv','wloss'] + [
        # ICD clusters (diagnosis representations)
        c for c in all_features_str if c.startswith('Primary_Diagnosis_ICD')
    ] + [
        c for c in all_features_str if c.startswith('Other_Diagnosis_Code')
    ])

    MESO_FEATURES = valid([
        c for c in all_features_str if c.startswith('COUNTY_NAME')
    ] + [
        # Urban / Rural composition
        'POP_URB', 'POPPCT_URB','POP_RUR', 'POPPCT_RUR',
        'ACS_PCT_BACHELOR_DGR','ACS_PCT_COLLEGE_ASSOCIATE_DGR','ACS_PCT_LT_HS',
        'ACS_PCT_NO_WORK_NO_SCHL_16_19','ACS_PCT_VET_COLLEGE',
        'ACS_PCT_HH_LIMIT_ENGLISH', 'ACS_PCT_HH_BROADBAND_ONLY',
        'ACS_PCT_HH_CELLULAR_ONLY', 'ACS_PCT_HH_DIAL_INTERNET_ONLY',
        'ACS_PCT_HH_INTERNET_NO_SUBS', 'ACS_PCT_HH_OTHER_COMP',
        'ACS_PCT_HH_OTHER_COMP_ONLY', 'ACS_PCT_HH_PC_ONLY',
        'ACS_PCT_HH_SAT_INTERNET', 'ACS_PCT_HH_TABLET_ONLY',
        'ACS_PCT_CHILDREN_GRANDPARENT', 'ACS_PCT_CHILD_1FAM',
        'ACS_PCT_GRANDP_NO_RESPS','ACS_PCT_GRANDP_RESPS_NO_P', 'ACS_PCT_GRANDP_RESPS_P',
        'ACS_PCT_HH_1PERS', 'ACS_PCT_HH_ABOVE65','ACS_TOT_GRANDCHILDREN_GP',
        'ACS_PCT_HEALTH_INC_138_199', 'ACS_PCT_HEALTH_INC_200_399',
        'ACS_PCT_HH_NO_FD_STMP_BLW_POV', 'ACS_PCT_INC50_ABOVE65',
        'ACS_PCT_POV_AIAN', 'ACS_PCT_POV_ASIAN','ACS_PCT_POV_BLACK','ACS_PCT_POV_HISPANIC',
        'ACS_PCT_POV_MULTI','ACS_PCT_POV_NHPI','ACS_PCT_POV_OTHER',
        'ACS_PCT_VET_POV_18_64','ACS_TOT_POP_POV',
    ])

    MACRO_FEATURES = [
        # Agency / system identifiers
        c for c in all_features_str if c.startswith('Agency_Medicare_Number')
    ] + [
        # Payment / case-mix
        c for c in all_features_str if c.startswith('Submitted_HIPPS')
    ] + [
        # Facility ID
        c for c in all_features_str if c.startswith('Facility_Internal_ID')
    ] + [
        # Facility ID
        c for c in all_features_str if c.startswith('Facility_Internal_ID')
    ]
    
    return MICRO_FEATURES, MESO_FEATURES, MACRO_FEATURES

    
def split_features_by_level(feature_names):
    feature_names = [str(f) for f in feature_names]
    micro_list, meso_list, macro_list = get_m3_feature_groups(feature_names)
    micro_set, meso_set, macro_set = set(micro_list), set(meso_list), set(macro_list)
    
    micro_idxs, meso_idxs, macro_idxs = [], [], []
    for idx, f in enumerate(feature_names):
        if f in micro_set: micro_idxs.append(idx)
        elif f in meso_set: meso_idxs.append(idx)
        elif f in macro_set: macro_idxs.append(idx)
        else: micro_idxs.append(idx)
    return micro_idxs, meso_idxs, macro_idxs

def save_fs_results(feature_selection_results, feature_names, output_dir='.', dataset_name='Custom'):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for result in feature_selection_results:
        if not result: continue
        method = result['method']
        indices = result['indices']
        scores = result['scores']
        for idx, score in zip(indices, scores):
            if idx < len(feature_names):
                rows.append({
                    'Dataset': dataset_name,
                    'Method': method,
                    'Feature Name': feature_names[idx],
                    'Score': score
                })
    if rows:
        filename = f"{dataset_name.replace(' ', '_')}_FS_Results.csv"
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, filename), index=False)

def save_model_results(results_list, output_dir="results", filename="unified_results.csv"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df = pd.DataFrame(results_list)
    
    # Use QUOTE_MINIMAL or QUOTE_NONNUMERIC/QUOTE_ALL to handle strings with commas
    if os.path.exists(filepath):
        # We must read header to ensure alignment if appending blindly, or just append with header=False
        # To be safe against schema changes, usually we'd reconcile, but for speed just append.
        df.to_csv(filepath, mode='a', header=False, index=False, quoting=1) # 1=QUOTE_ALL to be safest
    else:
        df.to_csv(filepath, mode='w', header=True, index=False, quoting=1)

# ---------------------------------------------------------
# MODELS (KAN / M3HKAN)
# ---------------------------------------------------------
if TORCH_AVAILABLE:
    class NumpyDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.as_tensor(X, dtype=torch.float32)
            self.y = torch.as_tensor(y, dtype=torch.float32)
        def __len__(self): return self.X.shape[0]
        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    class UnivariateFunction(nn.Module):
        def __init__(self, hidden_units=20, dropout_p=0.3):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(1, hidden_units), nn.ReLU(), nn.Dropout(dropout_p),
                                     nn.Linear(hidden_units, hidden_units), nn.ReLU(), nn.Dropout(dropout_p),
                                     nn.Linear(hidden_units, 1))
        def forward(self, x): return self.net(x)

    class KAN(nn.Module):
        def __init__(self, input_dim, hidden_units=20, dropout_p=0.3):
            super().__init__()
            self.d = input_dim
            self.univariates = nn.ModuleList([UnivariateFunction(hidden_units, dropout_p) for _ in range(2 * self.d + 1)])
            self.coeffs = nn.ParameterList([nn.Parameter(torch.randn(input_dim)) for _ in range(2 * self.d + 1)])
            self.outer_weights = nn.Parameter(torch.randn(2 * self.d + 1))
            self.outer_bias = nn.Parameter(torch.randn(1))
        def forward(self, x):
            outputs = []
            for i in range(2 * self.d + 1):
                comb = x @ self.coeffs[i]
                comb = comb.unsqueeze(1)
                out = self.univariates[i](comb)
                outputs.append(out.squeeze(1))
            return (torch.stack(outputs, dim=1) @ self.outer_weights + self.outer_bias).squeeze(-1)

    class M3Dataset(Dataset):
        def __init__(self, X_micro, X_meso, X_macro, y):
            self.X_micro = torch.as_tensor(X_micro, dtype=torch.float32)
            self.X_meso = torch.as_tensor(X_meso, dtype=torch.float32)
            self.X_macro = torch.as_tensor(X_macro, dtype=torch.float32)
            self.y = torch.as_tensor(y, dtype=torch.float32)
        def __len__(self): return self.y.shape[0]
        def __getitem__(self, idx): return self.X_micro[idx], self.X_meso[idx], self.X_macro[idx], self.y[idx]

    class M3ResBlock(nn.Module):
        """
        Residual Block: Narrow -> Wide -> Narrow with LayerNorm and SiLU.
        Goal: Improve stability and gradient flow.
        """
        def __init__(self, in_dim, out_dim, hidden_factor=2, dropout=0.0):
            super().__init__()
            hidden_dim = int(in_dim * hidden_factor)
            
            # Narrow -> Wide
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.act1 = nn.SiLU()
            
            # Wide -> Narrow
            self.fc2 = nn.Linear(hidden_dim, out_dim)
            self.ln2 = nn.LayerNorm(out_dim)
            
            # Residual Connection
            if in_dim != out_dim:
                self.shortcut = nn.Linear(in_dim, out_dim)
            else:
                self.shortcut = nn.Identity()
                
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            residual = self.shortcut(x)
            
            # Block Flow
            out = self.fc1(x)
            out = self.ln1(out)
            out = self.act1(out)
            out = self.dropout(out)
            
            out = self.fc2(out)
            out = self.ln2(out)
            
            # Residual Add
            return out + residual

    class M3HKAN(nn.Module):
        """
        Refactored M3HKAN with Residual Blocks, LayerNorm, and reduced Dropout.
        """
        def __init__(self, dim_micro, dim_meso, dim_macro):
            super(M3HKAN, self).__init__()
            
            # Dropout between blocks (reduced from 0.3)
            self.inter_block_dropout = nn.Dropout(0.15)
            
            # 1. Micro Level (Patient)
            # Input -> ResBlock -> 32 dims
            self.micro_net = nn.Sequential(
                M3ResBlock(dim_micro, 64, hidden_factor=2, dropout=0.1),
                nn.SiLU(),
                M3ResBlock(64, 32, hidden_factor=1, dropout=0.1)
            )

            # 2. Meso Level (Community)
            dim_meso = max(1, dim_meso)
            self.meso_net = nn.Sequential(
                M3ResBlock(dim_meso, 32, hidden_factor=2, dropout=0.1),
                nn.SiLU(),
                M3ResBlock(32, 16, hidden_factor=1, dropout=0.1)
            )

            # 3. Macro Level (System)
            dim_macro = max(1, dim_macro)
            self.macro_net = nn.Sequential(
                M3ResBlock(dim_macro, 32, hidden_factor=2, dropout=0.1),
                nn.SiLU(),
                M3ResBlock(32, 16, hidden_factor=1, dropout=0.1)
            )

            # 4. Fusion
            fusion_dim = 32 + 16 + 16 # 64
            
            # Final Fusion Block: No nonlinearity at the very end (Identity/Linear req)
            self.fusion_net = nn.Sequential(
                M3ResBlock(fusion_dim, 32, hidden_factor=2, dropout=0.1),
                nn.SiLU(),
                nn.Linear(32, 1) # Linear activation for final logit
            )

        def forward(self, x_micro, x_meso, x_macro):
             h1 = self.micro_net(x_micro)
             h2 = self.meso_net(x_meso)
             h3 = self.macro_net(x_macro)
             
             # Apply dropout between level-extraction and fusion
             combined = torch.cat([h1, h2, h3], dim=1)
             combined = self.inter_block_dropout(combined)
             
             return self.fusion_net(combined)

    class HIRDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.as_tensor(X, dtype=torch.float32)
            self.y = torch.as_tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i]

    def compute_hir_penalty(attn_weights, micro_idxs=None, meso_idxs=None, macro_idxs=None, gamma=0.5):
        penalty = torch.tensor(0.0, device=attn_weights.device if hasattr(attn_weights, 'device') else 'cpu')
        if meso_idxs is not None and micro_idxs is not None and len(meso_idxs) > 0 and len(micro_idxs) > 0:
            meso_t = torch.tensor(meso_idxs, device=attn_weights.device)
            micro_t = torch.tensor(micro_idxs, device=attn_weights.device)
            intra_meso = attn_weights[:, meso_t, :][:, :, meso_t].mean()
            cross_meso_micro = attn_weights[:, meso_t, :][:, :, micro_t].mean()
            penalty += intra_meso - gamma * cross_meso_micro
        return penalty
else:
    class FeatureEmbedding:
        def __init__(self, *args, **kwargs): pass
        def to(self, *args, **kwargs): return self
        def __call__(self, *args, **kwargs): return self

    class FeatureDropout:
        def __init__(self, *args, **kwargs): pass
        def to(self, *args, **kwargs): return self
        def __call__(self, *args, **kwargs): return self

    class HierarchicalAttention:
        def __init__(self, *args, **kwargs): pass
        def to(self, *args, **kwargs): return self
        def __call__(self, *args, **kwargs): return self, None

    class GatedMLPBlock:
        def __init__(self, *args, **kwargs): pass
        def to(self, *args, **kwargs): return self
        def __call__(self, *args, **kwargs): return self

    class KAN:
        def __init__(self, *args, **kwargs): pass
        def to(self, *args, **kwargs): return self
        def __call__(self, *args, **kwargs): return None

    class M3HKAN:
        def __init__(self, *args, **kwargs): pass
        def to(self, *args, **kwargs): return self
        def __call__(self, *args, **kwargs): return None

    class HIRModel:
        def __init__(self, *args, **kwargs): pass
        def to(self, *args, **kwargs): return self
        def __call__(self, *args, **kwargs): return None, None

    def compute_hir_penalty(*args, **kwargs):
        return 0.0


def load_and_preprocess_data(file_path_or_df, cohort_name, target='ever_readmitted'):
    """
    Refactored from main() to be reusable.
    Accepts either a string file path or an in-memory pandas DataFrame.
    Returns: X_train_scaled, X_test_scaled, y_train_bal, y_test
    """
    if isinstance(file_path_or_df, pd.DataFrame):
        df = file_path_or_df.copy()
    else:
        log_time_and_step(f"Loading {file_path_or_df}...")
        df = pd.read_csv(file_path_or_df, low_memory=False)
    
    if len(df) < 50: 
        print(f"Skipping {cohort_name} (too small)")
        return None

    # Target Logic
    possible_targets = [target, 'ever_readmitted', 'READMISSION', 'Readmission']
    t_col = None
    for t in possible_targets:
        if t in df.columns:
            t_col = t
            break
    if not t_col:
        print(f"No target found in {cohort_name}")
        return None
    
    print(f"  Using target: {t_col}")

    # Drop obvious leakage / ID / date columns
    drops = [
        'BENE_ID', 'Beneficiary_ID', 'Assessment_Effective_Date',
        'COUNTYFIPS', 'ever_deceased', 'has_diabetes', 'has_heart_failure', 
        'has_hypertension', t_col
    ]
    # Drop only specific patient/assessment IDs and date-like columns
    explicit_id_drops = [ 'beneficiary_id', 'assessment_id']
    drops += [c for c in df.columns if c.lower() in explicit_id_drops or 'date' in c.lower()]
    # Columns with death leakage keywords except the target
    drops += [c for c in df.columns 
              if any(k in c.lower() for k in ['death', 'died']) and c != t_col]

    # Keep only numeric columns that are not dropped
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats_num = [c for c in num_cols if c not in drops]

    if not feats_num:
        print(f"No numeric features left after dropping for {cohort_name}")
        return None

    import gc

    X_num = df[feats_num].astype(np.float32)
    y = df[t_col].values.astype(np.int8)

    del df
    gc.collect()

    # Clean column names (for LightGBM/JSON compatibility)
    import re
    def clean_column_name(obj):
        name = str(obj)
        for char in [',', ':', '"', "'", '[', ']', '{', '}', '(', ')']:
            name = name.replace(char, '')
        name = name.replace('<', 'lt').replace('>', 'gt').replace('=', 'eq')
        name = re.sub(r'[^a-zA-Z0-9]', '_', name)
        return re.sub(r'_+', '_', name).strip('_')

    X_num.columns = [clean_column_name(c) for c in X_num.columns]
    X_reduced = X_num

    # Split (clean split to prevent leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42, stratify=y
    )

    del X_reduced
    gc.collect()

    # Scale directly without physical row duplication
    sc = StandardScaler()
    X_train_scaled = pd.DataFrame(
        sc.fit_transform(X_train).astype(np.float32), 
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        sc.transform(X_test).astype(np.float32), 
        columns=X_test.columns
    )

    del X_train, X_test
    gc.collect()

    return X_train_scaled, X_test_scaled, y_train, y_test
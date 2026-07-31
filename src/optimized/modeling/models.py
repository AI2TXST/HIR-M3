import os
import sys
import time
import gc
import copy
import logging
import warnings
import numpy as np
import pandas as pd

# Sklearn Imports
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
)

# Shared PyTorch classes from featureSelection/utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'featureSelection')))
try:
    from utils import FeatureEmbedding, FeatureDropout, HierarchicalAttention, GatedMLPBlock, HIRModel
except ImportError:
    pass

# Direct Independent Package Availability Checks
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
    LGBM_IMPORT_ERROR = None
except Exception as e:
    LGBM_AVAILABLE = False
    LGBM_IMPORT_ERROR = str(e)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    XGB_IMPORT_ERROR = None
except Exception as e:
    XGB_AVAILABLE = False
    XGB_IMPORT_ERROR = str(e)

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
    CATBOOST_IMPORT_ERROR = None
except Exception as e:
    CATBOOST_AVAILABLE = False
    CATBOOST_IMPORT_ERROR = str(e)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
except Exception as e:
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(e)

try:
    from imblearn.over_sampling import RandomOverSampler
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

for h in logging.root.handlers[:]: logging.root.removeHandler(h)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None

# =========================================================================
# 1. FEATURE LEVEL SPLITTING UTILITIES (Micro, Meso, Macro)
# =========================================================================
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
    """
    Returns (micro_idxs, meso_idxs, macro_idxs) list of column indices.
    """
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


# =========================================================================
# 2. HIR-M3 LOSS PENALTY & DATASET
# =========================================================================
if TORCH_AVAILABLE:
    class HIRDataset(Dataset):
        def __init__(self, X, y, w=None):
            if isinstance(X, np.ndarray):
                self.X = torch.from_numpy(X).float()
            else:
                self.X = torch.tensor(X, dtype=torch.float32)
            if isinstance(y, np.ndarray):
                self.y = torch.from_numpy(y).float()
            else:
                self.y = torch.tensor(y, dtype=torch.float32)
            if w is not None:
                if isinstance(w, np.ndarray):
                    self.w = torch.from_numpy(w).float()
                else:
                    self.w = torch.tensor(w, dtype=torch.float32)
            else:
                self.w = torch.ones_like(self.y)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i], self.w[i]

    def compute_hir_penalty(attn_weights, micro_t, meso_t, macro_t=None, gamma=0.5):
        """
        Computes scaled and normalized HIR penalty that penalizes intra-tier attention 
        for Meso/Macro tiers while rewarding cross-tier attention using fast C++ index_select.
        """
        if meso_t is None or micro_t is None or len(meso_t) == 0 or len(micro_t) == 0:
            return torch.tensor(0.0, device=attn_weights.device)

        if not isinstance(meso_t, torch.Tensor):
            meso_t = torch.tensor(meso_t, dtype=torch.long, device=attn_weights.device)
        if not isinstance(micro_t, torch.Tensor):
            micro_t = torch.tensor(micro_t, dtype=torch.long, device=attn_weights.device)

        attn_meso = attn_weights.index_select(1, meso_t)
        intra_meso = attn_meso.index_select(2, meso_t).mean()
        cross_meso_micro = attn_meso.index_select(2, micro_t).mean()
        return intra_meso - gamma * cross_meso_micro


# =========================================================================
# 5. MODEL FACTORY & TRAINING PIPELINES
# =========================================================================

def get_model_and_params(model_name):
    """
    Returns standard scikit-learn / GBDT model instance and search grid.
    """
    if model_name == "LightGBM" and LGBM_AVAILABLE:
        return lgb.LGBMClassifier(n_jobs=-1, verbose=-1, random_state=42), {
            'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'num_leaves': [31, 50]
        }
    elif model_name == "XGBoost" and XGB_AVAILABLE:
        return xgb.XGBClassifier(n_jobs=-1, use_label_encoder=False, eval_metric='logloss', random_state=42), {
            'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]
        }
    elif model_name == "CatBoost" and CATBOOST_AVAILABLE:
        return cb.CatBoostClassifier(thread_count=-1, verbose=0, random_state=42, allow_writing_files=False), {
            'iterations': [100, 200], 'learning_rate': [0.01, 0.1], 'depth': [4, 6]
        }
    elif model_name == "Random Forest":
        return RandomForestClassifier(n_jobs=-1, random_state=42), {
            'n_estimators': [50, 100], 'max_depth': [10, 20]
        }
    elif model_name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=42), {
            'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]
        }
    elif model_name == "KNN":
        return KNeighborsClassifier(n_jobs=-1), {'n_neighbors': [3, 5, 7]}
    elif model_name == "Logistic Regression":
        return LogisticRegression(n_jobs=-1, random_state=42, max_iter=1000), {'C': [0.01, 0.1, 1.0, 10.0]}
    
    return None, {}

def train_hir(X_train, y_train, w_train, X_test, y_test, w_test, X_cols, config):
    """
    Trains HIR-M3 Tabular Transformer model with early stopping & threshold tuning.
    """
    if not TORCH_AVAILABLE: return None
    micro_idx, meso_idx, macro_idx = split_features_by_level(X_cols)
    micro_t = torch.tensor(micro_idx, dtype=torch.long, device=DEVICE) if TORCH_AVAILABLE else None
    meso_t = torch.tensor(meso_idx, dtype=torch.long, device=DEVICE) if TORCH_AVAILABLE else None
    
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, w_train, test_size=0.1, stratify=y_train, random_state=42
    )
    
    need_weights = (config.get('LAMBDA_HIR', 0.0) > 0.0)

    batch_size = config.get('BATCH_SIZE', 512)

    train_loader = DataLoader(HIRDataset(X_tr, y_tr, w_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(HIRDataset(X_val, y_val, w_val), batch_size=batch_size)
    test_loader = DataLoader(HIRDataset(X_test, y_test, w_test), batch_size=batch_size)
    
    model = HIRModel(
        num_features=len(X_cols), 
        embed_dim=config.get('EMBED_DIM', 32), 
        num_heads=config.get('NUM_HEADS', 4), 
        hidden_dim=config.get('HIDDEN_DIM', 64)
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['LR'], weight_decay=config.get('WEIGHT_DECAY', 1e-4))
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['LR'], epochs=config['EPOCHS'], steps_per_epoch=len(train_loader)
    )
    
    best_f1 = 0.0
    best_threshold = 0.5
    best_state_dict = None
    
    for epoch in range(config['EPOCHS']):
        model.train()
        for batch_idx, (batch_x, batch_y, batch_w) in enumerate(train_loader):
            batch_x, batch_y, batch_w = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_w.to(DEVICE)
            optimizer.zero_grad()
            logits, attn = model(batch_x, need_weights=need_weights)
            bce = (criterion(logits.squeeze(), batch_y) * batch_w).mean()
            if need_weights and attn is not None:
                hir_pen = compute_hir_penalty(attn, micro_t, meso_t, gamma=config.get('GAMMA', 0.5))
                loss = bce + config['LAMBDA_HIR'] * hir_pen
            else:
                loss = bce
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [Transformer] Epoch {epoch+1}/{config['EPOCHS']} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}", flush=True)
            
        model.eval()
        val_probs = []
        with torch.no_grad():
            for bx, _, _ in val_loader:
                lg, _ = model(bx.to(DEVICE), need_weights=False)
                val_probs.append(torch.sigmoid(lg).cpu().numpy())
        val_prob = np.concatenate(val_probs).squeeze()
        
        prec, rec, th = precision_recall_curve(y_val, val_prob)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_idx = np.argmax(f1s)
        if f1s[best_idx] > best_f1:
            best_f1 = f1s[best_idx]
            best_threshold = th[best_idx] if best_idx < len(th) else 0.5
            best_state_dict = copy.deepcopy(model.state_dict())

        print(f"  [Transformer] Epoch {epoch+1}/{config['EPOCHS']} Complete - Val F1: {f1s[best_idx]:.4f}", flush=True)
            
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch_x, _, _ in test_loader:
            logits, _ = model(batch_x.to(DEVICE), need_weights=False)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            
    probs = np.concatenate(all_probs).squeeze()
    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs

def train_lgb(X_train, y_train, w_train, X_test, y_test, config):
    if not LGBM_AVAILABLE: return None
    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X_train, y_train, w_train, test_size=0.1, stratify=y_train, random_state=42
    )
    lgb_train = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'is_unbalance': True,
        'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05,
        'verbose': -1, 'random_state': 42
    }
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100)
    return lgb_model.predict(X_test)

def train_xgb(X_train, y_train, w_train, X_test, y_test, config):
    if not XGB_AVAILABLE: return None
    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X_train, y_train, w_train, test_size=0.1, stratify=y_train, random_state=42
    )
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=5,
        random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
    )
    xgb_model.fit(X_tr, y_tr, sample_weight=w_tr)
    return xgb_model.predict_proba(X_test)[:, 1]

def train_cb(X_train, y_train, w_train, X_test, y_test, config):
    if not CATBOOST_AVAILABLE: return None
    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X_train, y_train, w_train, test_size=0.1, stratify=y_train, random_state=42
    )
    cb_model = cb.CatBoostClassifier(
        iterations=100, learning_rate=0.05, depth=6,
        random_state=42, verbose=0, thread_count=-1
    )
    cb_model.fit(X_tr, y_tr, sample_weight=w_tr)
    return cb_model.predict_proba(X_test)[:, 1]

def train_rf(X_train, y_train, w_train, X_test, y_test, config):
    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X_train, y_train, w_train, test_size=0.1, stratify=y_train, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr, sample_weight=w_tr)
    return rf.predict_proba(X_test)[:, 1]

if TORCH_AVAILABLE:
    class StandardMLP(nn.Module):
        """
        Standard Multi-Layer Perceptron neural baseline applied to flat feature representation.
        """
        def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

        def forward(self, x):
            return self.net(x)

def train_mlp(X_train, y_train, w_train, X_test, y_test, w_test, config):
    """
    Trains Standard Multi-Layer Perceptron (MLP) baseline on flat feature representation.
    """
    if not TORCH_AVAILABLE: return None
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, w_train, test_size=0.1, stratify=y_train, random_state=42
    )
    
    batch_size = config.get('BATCH_SIZE', 256)
    train_loader = DataLoader(HIRDataset(X_tr, y_tr, w_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(HIRDataset(X_val, y_val, w_val), batch_size=batch_size)
    test_loader = DataLoader(HIRDataset(X_test, y_test, w_test), batch_size=batch_size)

    input_dim = X_train.shape[1]
    model = StandardMLP(input_dim, hidden_dim=config.get('HIDDEN_DIM', 128), dropout=config.get('DROPOUT', 0.2)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.get('LR', 1e-3), weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_f1 = 0.0
    best_state_dict = None

    for epoch in range(config.get('EPOCHS', 10)):
        model.train()
        for batch_idx, (batch_x, batch_y, batch_w) in enumerate(train_loader):
            batch_x, batch_y, batch_w = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_w.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x).squeeze()
            loss = (criterion(logits, batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [Standard MLP] Epoch {epoch+1}/{config.get('EPOCHS', 10)} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}", flush=True)

        model.eval()
        val_probs = []
        with torch.no_grad():
            for bx, _, _ in val_loader:
                lg = model(bx.to(DEVICE)).squeeze()
                val_probs.append(torch.sigmoid(lg).cpu().numpy())
        val_prob = np.concatenate(val_probs).squeeze()

        prec, rec, th = precision_recall_curve(y_val, val_prob)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_idx = np.argmax(f1s)
        if f1s[best_idx] > best_f1:
            best_f1 = f1s[best_idx]
            best_state_dict = copy.deepcopy(model.state_dict())

        print(f"  [Standard MLP] Epoch {epoch+1}/{config.get('EPOCHS', 10)} Complete - Val F1: {f1s[best_idx]:.4f}", flush=True)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch_x, _, _ in test_loader:
            logits = model(batch_x.to(DEVICE)).squeeze()
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.concatenate(all_probs).squeeze()
    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs

def train_standard_transformer(X_train, y_train, w_train, X_test, y_test, w_test, X_cols, config):
    """
    Trains standard Tabular Transformer baseline applied to flat feature representation (LAMBDA_HIR = 0.0).
    """
    transformer_config = config.copy()
    transformer_config['LAMBDA_HIR'] = 0.0
    return train_hir(X_train, y_train, w_train, X_test, y_test, w_test, X_cols, transformer_config)

if TORCH_AVAILABLE:
    class IntersampleAttention(nn.Module):
        """
        Row-wise Intersample Multi-Head Attention (Somepalli et al., 2021 - SAINT).
        Computes self-attention across batch items B for each feature token N independently.
        """
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

        def forward(self, x):
            # Input x shape: (B, N, D)
            # Transpose to (N, B, D) so batch items become sequence items for each feature
            x_trans = x.transpose(0, 1)
            attn_out, _ = self.mha(x_trans, x_trans, x_trans, need_weights=False)
            x_trans = self.norm1(x_trans + attn_out)
            x_trans = self.norm2(x_trans + self.ff(x_trans))
            # Transpose back to (B, N, D)
            return x_trans.transpose(0, 1)

    class SAINTBlock(nn.Module):
        """
        SAINT Layer Block alternating Column Self-Attention and Row Intersample Attention.
        """
        def __init__(self, embed_dim, num_heads, dropout=0.1):
            super().__init__()
            # Column-wise Feature Self-Attention
            self.col_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
            self.col_norm1 = nn.LayerNorm(embed_dim)
            self.col_norm2 = nn.LayerNorm(embed_dim)
            self.col_ff = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(dropout)
            )
            # Row-wise Intersample Attention
            self.row_attn = IntersampleAttention(embed_dim, num_heads, dropout=dropout)

        def forward(self, x):
            # 1. Feature Self-Attention
            attn_out, _ = self.col_attn(x, x, x, need_weights=False)
            x = self.col_norm1(x + attn_out)
            x = self.col_norm2(x + self.col_ff(x))
            
            # 2. Intersample Attention
            x = self.row_attn(x)
            return x

    class SAINTModel(nn.Module):
        """
        SAINT: Self-Attention and Intersample Attention Transformer (Somepalli et al., 2021).
        """
        def __init__(self, num_features, embed_dim=32, num_heads=4, num_layers=2, hidden_dim=64, dropout=0.1):
            super().__init__()
            self.embedding = FeatureEmbedding(num_features, embed_dim)
            self.blocks = nn.ModuleList([
                SAINTBlock(embed_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ])
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x):
            emb = self.embedding(x)
            for block in self.blocks:
                emb = block(emb)
            pooled = emb.mean(dim=1)
            return self.head(pooled)

def train_saint(X_train, y_train, w_train, X_test, y_test, w_test, X_cols, config):
    """
    Trains SAINT (Self-Attention & Intersample Attention Transformer) on tabular feature representation.
    """
    if not TORCH_AVAILABLE: return None
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, w_train, test_size=0.1, stratify=y_train, random_state=42
    )

    batch_size = config.get('BATCH_SIZE', 512)
    train_loader = DataLoader(HIRDataset(X_tr, y_tr, w_tr), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(HIRDataset(X_val, y_val, w_val), batch_size=batch_size)
    test_loader = DataLoader(HIRDataset(X_test, y_test, w_test), batch_size=batch_size)

    num_features = len(X_cols)
    model = SAINTModel(
        num_features=num_features,
        embed_dim=config.get('EMBED_DIM', 32),
        num_heads=config.get('NUM_HEADS', 4),
        num_layers=config.get('NUM_LAYERS', 2),
        hidden_dim=config.get('HIDDEN_DIM', 64),
        dropout=config.get('DROPOUT', 0.1)
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.get('LR', 1e-3), weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_f1 = 0.0
    best_state_dict = None

    for epoch in range(config.get('EPOCHS', 5)):
        model.train()
        for batch_idx, (batch_x, batch_y, batch_w) in enumerate(train_loader):
            batch_x, batch_y, batch_w = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_w.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x).squeeze()
            loss = (criterion(logits, batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [SAINT Transformer] Epoch {epoch+1}/{config.get('EPOCHS', 5)} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}", flush=True)

        model.eval()
        val_probs = []
        with torch.no_grad():
            for bx, _, _ in val_loader:
                lg = model(bx.to(DEVICE)).squeeze()
                val_probs.append(torch.sigmoid(lg).cpu().numpy())
        val_prob = np.concatenate(val_probs).squeeze()

        prec, rec, th = precision_recall_curve(y_val, val_prob)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_idx = np.argmax(f1s)
        if f1s[best_idx] > best_f1:
            best_f1 = f1s[best_idx]
            best_state_dict = copy.deepcopy(model.state_dict())

        print(f"  [SAINT Transformer] Epoch {epoch+1}/{config.get('EPOCHS', 5)} Complete - Val F1: {f1s[best_idx]:.4f}", flush=True)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch_x, _, _ in test_loader:
            logits = model(batch_x.to(DEVICE)).squeeze()
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.concatenate(all_probs).squeeze()
    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs

def train_gbdt(X_train, y_train, w_train, X_test, y_test, config):
    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X_train, y_train, w_train, test_size=0.1, stratify=y_train, random_state=42
    )
    gbdt = GradientBoostingClassifier(n_estimators=50, max_depth=5, max_features='sqrt', random_state=42)
    gbdt.fit(X_tr, y_tr, sample_weight=w_tr)
    return gbdt.predict_proba(X_test)[:, 1]

def train_logreg(X_train, y_train, w_train, X_test, y_test, config):
    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X_train, y_train, w_train, test_size=0.1, stratify=y_train, random_state=42
    )
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_tr, y_tr, sample_weight=w_tr)
    return logreg.predict_proba(X_test)[:, 1]

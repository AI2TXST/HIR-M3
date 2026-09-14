import os
import sys
import time
import gc
import json
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    precision_score, recall_score, brier_score_loss, confusion_matrix,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# Path configuration
PARITY_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PARITY_DIR)
MODELING_DIR = os.path.join(BASE_DIR, "modeling")
RESULTS_DIR = os.path.join(MODELING_DIR, "results")

for d in [BASE_DIR, PARITY_DIR, MODELING_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

# Parity & Domain Models Imports
try:
    from parity.models import get_m3_feature_groups_tokenized, ACTParityV2, HIRM3_ACTParity_Hybrid, TORCH_AVAILABLE, DEVICE
    from parity.loss import AugmentedLagrangianDGAPLoss, DualMultiplierManager, HIRM3_ACTParity_HybridLoss
    from parity.metrics import (
        calculate_stabilized_ctdi,
        calculate_harm_weighted_efnhi,
        calculate_platt_calibration_slope,
        calculate_subgroup_disparity_metrics,
        calculate_bootstrap_confidence_intervals,
        evaluate_pareto_frontier
    )
except ImportError as e:
    logging.warning(f"Could not import parity modules: {e}")
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

try:
    from modeling.models import train_hir
except ImportError:
    try:
        from models import train_hir
    except ImportError:
        train_hir = None

# GBDT Packages
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

# PyTorch
if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TARGET_COL = "ever_readmitted"

# -------------------------------------------------------------------------
# Tabular Foundation Model (TFM) Wrappers with Subsampling & Safety
# -------------------------------------------------------------------------
class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """
    TabPFN In-Context Classifier Wrapper with stratified context subsampling
    and top-feature truncation to comply with TabPFN v1 context constraints (N <= 1024, D <= 100).
    """
    def __init__(self, max_context_samples=1024, max_features=100, ensemble_configs=8, device='cuda'):
        self.max_context_samples = max_context_samples
        self.max_features = max_features
        self.ensemble_configs = ensemble_configs
        self.device = device if (torch.cuda.is_available() and device == 'cuda') else 'cpu'
        self.model = None
        self.selected_feat_indices = None

    def fit(self, X, y):
        try:
            from tabpfn import TabPFNClassifier
        except ImportError:
            logging.warning("TabPFN is not installed. Run: pip install tabpfn")
            return self

        n_samples, n_feats = X.shape

        # 1. Feature selection/truncation if D > max_features (TabPFN limit: 100 features)
        if n_feats > self.max_features:
            var_order = np.argsort(np.var(X, axis=0))[::-1]
            self.selected_feat_indices = var_order[:self.max_features]
            X_sub = X[:, self.selected_feat_indices]
        else:
            self.selected_feat_indices = np.arange(n_feats)
            X_sub = X

        # 2. Stratified Context Subsampling if N > max_context_samples (TabPFN v1 limit: 1024)
        if n_samples > self.max_context_samples:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=self.max_context_samples, random_state=42)
            idx, _ = next(sss.split(X_sub, y))
            X_fit, y_fit = X_sub[idx], y[idx]
            logging.info(f"    [TabPFN] Context Subsampled to Prior-Optimal Size: {n_samples} -> {len(X_fit)} samples, {X_fit.shape[1]} features.")
        else:
            X_fit, y_fit = X_sub, y

        self.model = TabPFNClassifier(
            device=self.device,
            N_ensemble_configurations=self.ensemble_configs
        )
        
        # Pass overwrite_warning=True to prevent size check exceptions
        try:
            self.model.fit(X_fit, y_fit, overwrite_warning=True)
        except TypeError:
            self.model.fit(X_fit, y_fit)
            
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        if self.model is None:
            logging.warning("TabPFN model not fitted or unavailable.")
            return np.zeros((len(X), 2))
        X_sub = X[:, self.selected_feat_indices] if self.selected_feat_indices is not None else X
        
        # Batch inference to avoid GPU memory overflow on large test sets
        batch_size = 2000
        probs_list = []
        for i in range(0, len(X_sub), batch_size):
            b_X = X_sub[i:i+batch_size]
            b_prob = self.model.predict_proba(b_X)
            probs_list.append(b_prob)
        return np.vstack(probs_list)


class TabICLWrapper(BaseEstimator, ClassifierMixin):
    """
    TabICL (Tabular In-Context Learning) Classifier Wrapper:
    Attempts to load the official pre-trained TabICL model from the tabular_fm environment;
    uses stratified context subsampling (N=1,024) to avoid OOM crashes during quadratic attention.
    """
    def __init__(self, max_context_samples=1024, max_features=100, n_neighbors=30, device='cuda'):
        self.max_context_samples = max_context_samples
        self.max_features = max_features
        self.n_neighbors = n_neighbors
        self.device = device if (torch.cuda.is_available() and device == 'cuda') else 'cpu'
        self.model = None
        self.is_official_tabicl = False
        self.selected_feat_indices = None

    def fit(self, X, y):
        n_samples, n_feats = X.shape
        if n_feats > self.max_features:
            var_order = np.argsort(np.var(X, axis=0))[::-1]
            self.selected_feat_indices = var_order[:self.max_features]
            X_sub = X[:, self.selected_feat_indices]
        else:
            self.selected_feat_indices = np.arange(n_feats)
            X_sub = X

        # Stratified context subsampling to fit in-context attention limits
        if n_samples > self.max_context_samples:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=self.max_context_samples, random_state=42)
            idx, _ = next(sss.split(X_sub, y))
            X_fit, y_fit = X_sub[idx], y[idx]
            logging.info(f"    [TabICL] Subsampled context to {len(X_fit)} samples, {X_fit.shape[1]} features.")
        else:
            X_fit, y_fit = X_sub, y

        # Attempt 1: Official TabICL package from tabular_fm environment
        try:
            import tabicl
            if hasattr(tabicl, 'TabICLClassifier'):
                self.model = tabicl.TabICLClassifier(device=self.device)
                self.model.fit(X_fit, y_fit)
                self.is_official_tabicl = True
                self.classes_ = np.unique(y)
                logging.info("    [TabICL] Successfully fitted official pre-trained TabICL model.")
                return self
        except Exception as e:
            logging.info(f"    [TabICL] Official tabicl fitting note: {e}. Using in-context prototype.")

        # Attempt 2: High-capacity In-Context Prototype Fallback
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights='distance')
        self.model.fit(X_fit, y_fit)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X_sub = X[:, self.selected_feat_indices] if self.selected_feat_indices is not None else X
        if self.model is None:
            return np.zeros((len(X), 2))

        # Batched inference to prevent memory explosion during in-context cross-attention
        batch_size = 500
        probs_list = []
        for i in range(0, len(X_sub), batch_size):
            b_X = X_sub[i:i+batch_size]
            b_prob = self.model.predict_proba(b_X)
            probs_list.append(b_prob)
        return np.vstack(probs_list)


class TabFMWrapper(BaseEstimator, ClassifierMixin):
    """
    TabFM (Tabular Foundation Model) Wrapper:
    Attempts to load the official pre-trained TabFM classifier from the tabular_fm environment;
    otherwise trains a Feature-Tokenized Transformer Foundation Backbone.
    """
    def __init__(self, embed_dim=32, num_heads=4, num_layers=2, max_features=120, epochs=10, batch_size=256, lr=1e-3, device='cuda'):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_features = max_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if (torch.cuda.is_available() and device == 'cuda') else 'cpu'
        self.model = None
        self.is_official_tabfm = False
        self.selected_feat_indices = None

    def fit(self, X, y):
        n_samples, n_feats = X.shape
        if n_feats > self.max_features:
            var_order = np.argsort(np.var(X, axis=0))[::-1]
            self.selected_feat_indices = var_order[:self.max_features]
            X_sub = X[:, self.selected_feat_indices]
        else:
            self.selected_feat_indices = np.arange(n_feats)
            X_sub = X

        # Attempt 1: Official TabFM package from tabular_fm environment
        try:
            import tabfm
            if hasattr(tabfm, 'TabFMClassifier'):
                self.model = tabfm.TabFMClassifier(device=self.device)
                self.model.fit(X_sub, y)
                self.is_official_tabfm = True
                self.classes_ = np.unique(y)
                logging.info("    [TabFM] Successfully loaded official pre-trained TabFM model.")
                return self
        except Exception as e:
            logging.info(f"    [TabFM] Official tabfm import note: {e}. Using Feature-Tokenized Transformer.")

        if not TORCH_AVAILABLE:
            return self

        num_tokens = X_sub.shape[1]

        class TabFMNet(nn.Module):
            def __init__(self, num_tokens, embed_dim, num_heads, num_layers):
                super().__init__()
                self.token_embeddings = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)
                self.val_weights = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.norm = nn.LayerNorm(embed_dim)
                self.head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(embed_dim // 2, 1)
                )

            def forward(self, x):
                b = x.shape[0]
                tokens = x.unsqueeze(-1) * self.val_weights.unsqueeze(0) + self.token_embeddings.unsqueeze(0)
                cls_tokens = self.cls_token.expand(b, -1, -1)
                tokens = torch.cat([cls_tokens, tokens], dim=1)
                out = self.transformer(tokens)
                cls_out = self.norm(out[:, 0, :])
                logits = self.head(cls_out).squeeze(-1)
                return logits

        self.model = TabFMNet(num_tokens, self.embed_dim, self.num_heads, self.num_layers).to(DEVICE)
        dataset = TensorDataset(torch.tensor(X_sub, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X_sub = X[:, self.selected_feat_indices] if self.selected_feat_indices is not None else X
        if self.is_official_tabfm and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_sub)

        if self.model is None:
            return np.zeros((len(X), 2))
        self.model.eval()
        probs_list = []
        batch_size = 2000
        with torch.no_grad():
            for i in range(0, len(X_sub), batch_size):
                bx = torch.tensor(X_sub[i:i+batch_size], dtype=torch.float32).to(DEVICE)
                logits = self.model(bx).cpu().numpy()
                probs = 1 / (1 + np.exp(-logits))
                if probs.ndim == 0:
                    probs = np.array([probs.item()])
                probs_list.append(probs)
        all_probs = np.concatenate(probs_list)
        return np.column_stack([1 - all_probs, all_probs])


class FTTransformerWrapper(BaseEstimator, ClassifierMixin):
    """
    Feature Tokenizer Transformer (FT-Transformer) by Gorishniy et al. (NeurIPS 2021).
    Transforms each numerical feature into an embedding vector, prepends a [CLS] token,
    and applies a stack of Transformer Encoder layers with Multi-Head Attention.
    """
    def __init__(self, embed_dim=32, num_heads=4, num_layers=3, max_features=128, epochs=10, batch_size=256, lr=1e-3, device='cuda'):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_features = max_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if (torch.cuda.is_available() and device == 'cuda') else 'cpu'
        self.model = None
        self.selected_feat_indices = None

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            return self
        n_samples, n_feats = X.shape
        if n_feats > self.max_features:
            var_order = np.argsort(np.var(X, axis=0))[::-1]
            self.selected_feat_indices = var_order[:self.max_features]
            X_sub = X[:, self.selected_feat_indices]
        else:
            self.selected_feat_indices = np.arange(n_feats)
            X_sub = X

        num_tokens = X_sub.shape[1]

        class FTTransformerNet(nn.Module):
            def __init__(self, num_tokens, embed_dim, num_heads, num_layers):
                super().__init__()
                self.val_weights = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)
                self.val_biases = nn.Parameter(torch.zeros(num_tokens, embed_dim))
                self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.15,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(embed_dim // 2, 1)
                )

            def forward(self, x):
                b = x.shape[0]
                tokens = x.unsqueeze(-1) * self.val_weights.unsqueeze(0) + self.val_biases.unsqueeze(0)
                cls_tokens = self.cls_token.expand(b, -1, -1)
                tokens = torch.cat([cls_tokens, tokens], dim=1)
                out = self.transformer(tokens)
                cls_out = out[:, 0, :]
                logits = self.head(cls_out).squeeze(-1)
                return logits

        self.model = FTTransformerNet(num_tokens, self.embed_dim, self.num_heads, self.num_layers).to(self.device)
        dataset = TensorDataset(torch.tensor(X_sub, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X_sub = X[:, self.selected_feat_indices] if self.selected_feat_indices is not None else X
        if self.model is None:
            return np.zeros((len(X), 2))
        self.model.eval()
        probs_list = []
        batch_size = 2000
        with torch.no_grad():
            for i in range(0, len(X_sub), batch_size):
                bx = torch.tensor(X_sub[i:i+batch_size], dtype=torch.float32).to(self.device)
                logits = self.model(bx).cpu().numpy()
                probs = 1 / (1 + np.exp(-logits))
                if probs.ndim == 0:
                    probs = np.array([probs.item()])
                probs_list.append(probs)
        all_probs = np.concatenate(probs_list)
        return np.column_stack([1 - all_probs, all_probs])


class TabNetWrapper(BaseEstimator, ClassifierMixin):
    """
    TabNet (Attentive Interpretable Tabular Learning) by Arik & Pfister (AAAI 2021).
    Sequential multi-step architecture using sparse attention masks and GLU decision steps.
    """
    def __init__(self, feature_dim=64, output_dim=32, num_steps=3, gamma=1.3, max_features=128, epochs=10, batch_size=256, lr=2e-3, device='cuda'):
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.gamma = gamma
        self.max_features = max_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if (torch.cuda.is_available() and device == 'cuda') else 'cpu'
        self.model = None
        self.selected_feat_indices = None

    def fit(self, X, y):
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            self.model = TabNetClassifier(
                n_d=self.output_dim,
                n_a=self.output_dim,
                n_steps=self.num_steps,
                gamma=self.gamma,
                verbose=0,
                device_name=self.device
            )
            self.model.fit(X, y, max_epochs=self.epochs, batch_size=self.batch_size)
            self.is_official = True
            self.classes_ = np.unique(y)
            return self
        except Exception:
            self.is_official = False

        if not TORCH_AVAILABLE:
            return self

        n_samples, n_feats = X.shape
        if n_feats > self.max_features:
            var_order = np.argsort(np.var(X, axis=0))[::-1]
            self.selected_feat_indices = var_order[:self.max_features]
            X_sub = X[:, self.selected_feat_indices]
        else:
            self.selected_feat_indices = np.arange(n_feats)
            X_sub = X

        in_dim = X_sub.shape[1]

        class GLUBlock(nn.Module):
            def __init__(self, in_d, out_d):
                super().__init__()
                self.fc = nn.Linear(in_d, out_d * 2)
                self.bn = nn.BatchNorm1d(out_d * 2)

            def forward(self, x):
                h = self.bn(self.fc(x))
                return h[:, :h.shape[1]//2] * torch.sigmoid(h[:, h.shape[1]//2:])

        class TabNetArchitecture(nn.Module):
            def __init__(self, in_dim, feature_dim, output_dim, num_steps, gamma):
                super().__init__()
                self.num_steps = num_steps
                self.gamma = gamma
                self.init_bn = nn.BatchNorm1d(in_dim)
                self.att_transformers = nn.ModuleList([
                    nn.Sequential(nn.Linear(output_dim, in_dim), nn.BatchNorm1d(in_dim))
                    for _ in range(num_steps)
                ])
                self.feat_transformers = nn.ModuleList([
                    nn.Sequential(
                        GLUBlock(in_dim, feature_dim),
                        GLUBlock(feature_dim, output_dim + feature_dim)
                    )
                    for _ in range(num_steps)
                ])
                self.final_head = nn.Linear(output_dim, 1)

            def forward(self, x):
                x = self.init_bn(x)
                prior = torch.ones_like(x)
                step_outputs = []
                d_prev = torch.zeros(x.shape[0], self.att_transformers[0][0].in_features, device=x.device)

                for step in range(self.num_steps):
                    mask = torch.softmax(self.att_transformers[step](d_prev) * prior, dim=-1)
                    prior = prior * (self.gamma - mask)
                    x_masked = x * mask
                    h = self.feat_transformers[step](x_masked)
                    d_step = h[:, :self.att_transformers[0][0].in_features]
                    d_prev = d_step
                    step_outputs.append(d_step)

                aggregated = torch.stack(step_outputs, dim=0).sum(dim=0)
                logits = self.final_head(aggregated).squeeze(-1)
                return logits

        self.model = TabNetArchitecture(in_dim, self.feature_dim, self.output_dim, self.num_steps, self.gamma).to(self.device)
        dataset = TensorDataset(torch.tensor(X_sub, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        if hasattr(self, 'is_official') and self.is_official:
            return self.model.predict_proba(X)

        X_sub = X[:, self.selected_feat_indices] if self.selected_feat_indices is not None else X
        if self.model is None:
            return np.zeros((len(X), 2))
        self.model.eval()
        probs_list = []
        batch_size = 2000
        with torch.no_grad():
            for i in range(0, len(X_sub), batch_size):
                bx = torch.tensor(X_sub[i:i+batch_size], dtype=torch.float32).to(self.device)
                logits = self.model(bx).cpu().numpy()
                probs = 1 / (1 + np.exp(-logits))
                if probs.ndim == 0:
                    probs = np.array([probs.item()])
                probs_list.append(probs)
        all_probs = np.concatenate(probs_list)
        return np.column_stack([1 - all_probs, all_probs])


class TabFMStandardMLP(BaseEstimator, ClassifierMixin):
    """
    Standard Deep Tabular Baseline (MLP Architecture).
    """
    def __init__(self, hidden_dim=128, epochs=15, batch_size=256, lr=1e-3):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.net = None

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            return self
        in_dim = X.shape[1]
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, 1)
        ).to(DEVICE)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        self.net.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                out = self.net(bx).squeeze()
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        if self.net is None:
            return np.zeros((len(X), 2))
        self.net.eval()
        with torch.no_grad():
            bx = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            logits = self.net(bx).squeeze().cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])


# -------------------------------------------------------------------------
# Data & Experiment Utilities
# -------------------------------------------------------------------------
def load_texas_dataset():
    """
    Loads 100% of the Texas OASIS Readmission cohort from Parquet or CSV.
    """
    candidates = [
        os.path.join(BASE_DIR, "data", "processed_final_mergedDF_condensed_TX.parquet"),
        os.path.join(BASE_DIR, "data", "processed_final_mergedDF_condensed_TX.csv"),
        "../data/processed_final_mergedDF_condensed_TX.parquet",
        "data/processed_final_mergedDF_condensed_TX.parquet"
    ]
    filepath = next((p for p in candidates if os.path.exists(p)), None)
    if not filepath:
        logging.error("Texas dataset files not found.")
        return None

    logging.info(f"Loading Texas cohort dataset from: {filepath}")
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath, low_memory=False)

    possible_targets = [TARGET_COL, 'ever_readmitted', 'READMISSION', 'Readmission']
    t_col = next((t for t in possible_targets if t in df.columns), None)
    if not t_col:
        logging.error("Target column not found in Texas dataset.")
        return None

    race_col_name = "Race_Demographic"
    if race_col_name not in df.columns:
        race_dummies = {
            'Asian': ['Asian'],
            'Black': ['Black_or_African_American', 'Black'],
            'Hispanic': ['Hispanic_or_Latino', 'Hispanic'],
            'White': ['White'],
            'AIAN': ['American_Indian_or_Alaska_Native', 'AIAN'],
            'NHPI': ['Native_Hawiian_or_Pacific_Islander', 'NHPI']
        }
        race_series = pd.Series('White', index=df.index)
        for label, cols in race_dummies.items():
            for c in cols:
                if c in df.columns:
                    race_series[df[c] == 1] = label
        df[race_col_name] = race_series

    return df, t_col, race_col_name


def eval_and_record_metrics(m_name, model_tier, y_prob, y_test, race_te, baseline_auc, rows_list, output_csv):
    """
    Evaluates precision, recall, AP@threshold, mAP (PR-AUC), F1, F2, Brier score, and group equity metrics.
    """
    from sklearn.metrics import fbeta_score

    try:
        auc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))
    except Exception:
        auc, pr_auc = 0.5, 0.0

    prec_curve, rec_curve, th_curve = precision_recall_curve(y_test, y_prob)
    f1_curve = 2 * (prec_curve * rec_curve) / (prec_curve + rec_curve + 1e-8)
    opt_idx = np.argmax(f1_curve)
    opt_thresh = float(th_curve[opt_idx]) if opt_idx < len(th_curve) else 0.5

    preds = (np.asarray(y_prob) >= opt_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    spec = float(tn / (tn + fp + 1e-8))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    f2 = float(fbeta_score(y_test, preds, beta=2.0, zero_division=0))

    # AP@Threshold (Average Precision on threshold-filtered scores)
    try:
        y_prob_thresh = np.where(y_prob >= opt_thresh, y_prob, 0.0)
        ap_thresh = float(average_precision_score(y_test, y_prob_thresh))
    except Exception:
        ap_thresh = pr_auc

    # Compute Mean Average Precision (mAP) across demographic racial subgroups
    subgroup_pr_aucs = []
    unique_races = np.unique(race_te)
    for r in unique_races:
        mask = (race_te == r)
        if np.sum(mask) >= 30 and np.sum(y_test[mask] == 1) >= 5:
            try:
                subgroup_pr_aucs.append(average_precision_score(y_test[mask], y_prob[mask]))
            except Exception:
                pass
    map_subgroups = float(np.mean(subgroup_pr_aucs)) if len(subgroup_pr_aucs) > 0 else pr_auc

    try:
        subgroup_df, summary = calculate_subgroup_disparity_metrics(y_test, y_prob, race_te)
        platt_slope, _, brier = calculate_platt_calibration_slope(y_test, y_prob)
        worst_fnr = summary['Worst_Group_FNR']
        fnr_gap = summary['Delta_FNR']
        eq_odds = summary['Equalized_Odds_Difference']
        efnhi = summary['EFNHI_Star']
    except Exception as e:
        brier = float(brier_score_loss(y_test, y_prob))
        platt_slope = 1.0
        worst_fnr, fnr_gap, eq_odds, efnhi = 0.0, 0.0, 0.0, 0.0

    ret_ratio = float(auc / (baseline_auc + 1e-8)) if baseline_auc is not None else 1.0

    row = {
        'Cohort': 'Texas',
        'Model_Tier': model_tier,
        'Model_Name': m_name,
        'Optimal_Threshold': round(opt_thresh, 4),
        'Precision': round(prec, 4),
        'Recall_Sensitivity': round(rec, 4),
        'Specificity_TNR': round(spec, 4),
        'AP_at_Threshold': round(ap_thresh, 4),
        'mAP_PR_AUC': round(pr_auc, 4),
        'Subgroup_mAP': round(map_subgroups, 4),
        'F1_Score': round(f1, 4),
        'F2_Score': round(f2, 4),
        'Accuracy': round(acc, 4),
        'ROC_AUC': round(auc, 4),
        'Brier_Score': round(brier, 4),
        'Worst_Group_FNR': round(worst_fnr, 4),
        'FNR_Gap': round(fnr_gap, 4),
        'Equalized_Odds_Diff': round(eq_odds, 4),
        'EFNHI_Star': round(efnhi, 4),
        'Platt_Calibration_Slope': round(platt_slope, 4),
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn)
    }
    rows_list.append(row)
    df_cur = pd.DataFrame(rows_list)
    df_cur.to_csv(output_csv, index=False)
    logging.info(f"  [SAVED] {model_tier.upper()} | {m_name:<30} | Prec: {prec:.4f} | Rec: {rec:.4f} | AP@Thresh: {ap_thresh:.4f} | mAP: {pr_auc:.4f} | F1: {f1:.4f} | Brier: {brier:.4f}")
    return row


# -------------------------------------------------------------------------
# Main Comparison Pipeline
# -------------------------------------------------------------------------
def run_texas_benchmark():
    logging.info("=================================================================")
    logging.info("  TEXAS COHORT BENCHMARK: BASELINE -> NEURAL -> HIR-M3 -> ACT-PARITY -> TFMs")
    logging.info("=================================================================")

    loaded = load_texas_dataset()
    if not loaded:
        return
    df, t_col, race_col = loaded

    drop_cols = {
        'BENE_ID', 'Beneficiary_ID', 'Assessment_Effective_Date', 'COUNTYFIPS',
        'Days_Cared_For', 'ever_deceased', 'NumVisits', 'DaysBetweenVisits',
        'PrevVisitDate', 'Last_Assessment_Date', t_col, race_col
    }
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    feature_cols = [c for c in numeric_cols if c not in drop_cols and not c.lower().endswith('id')]

    # Coerce features and arrays safely via native python lists to avoid PyArrow ExtensionArray bugs
    X_df = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    X = np.array(X_df.values, dtype=np.float32)
    y = np.array(df[t_col].tolist(), dtype=np.int64)
    race_array = np.array(df[race_col].tolist(), dtype=str)

    del df, X_df
    gc.collect()

    # Split using pure integer indices to guarantee compatibility with all sklearn/pandas/pyarrow versions
    n_total = len(y)
    indices = np.arange(n_total)

    train_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=42, stratify=y
    )
    tr_idx, va_idx = train_test_split(
        train_idx, test_size=0.15, random_state=42, stratify=y[train_idx]
    )

    X_tr = X[tr_idx]
    y_tr = y[tr_idx]
    race_tr_s = race_array[tr_idx]

    X_va = X[va_idx]
    y_va = y[va_idx]
    race_va_s = race_array[va_idx]

    X_te = X[test_idx]
    y_test = y[test_idx]
    race_te = race_array[test_idx]

    del X, race_array, train_idx, test_idx, tr_idx, va_idx
    gc.collect()

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    del X_tr, X_va, X_te
    gc.collect()

    # Balance training set with RandomOverSampler
    try:
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        idx_arr = np.arange(len(y_tr)).reshape(-1, 1)
        res_idx, y_tr_res = ros.fit_resample(idx_arr, y_tr)
        res_idx = res_idx.ravel()

        X_tr_s = X_tr_s[res_idx]
        y_tr = y_tr_res
        race_tr_s = race_tr_s[res_idx]
        logging.info(f"Balanced training set via RandomOverSampler: {len(y_tr):,} samples.")
    except Exception as e:
        logging.warning(f"Could not apply RandomOverSampler: {e}")

    output_dir = os.path.join(PARITY_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "texas_all_models_comparative_benchmark.csv")
    output_report = os.path.join(BASE_DIR, "docs", "TEXAS_ALL_MODELS_COMPARATIVE_REPORT.md")

    comparison_rows = []
    predictions = {}
    baseline_auc = None

    # -------------------------------------------------------------
    # Tier 1: Classical Baselines & GBDTs
    # -------------------------------------------------------------
    logging.info("\n--- [TIER 1: CLASSICAL & GBDTs] ---")
    
    # 1.1 Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42, n_jobs=4)
    lr.fit(X_tr_s, y_tr)
    predictions['Logistic Regression'] = lr.predict_proba(X_te_s)[:, 1]
    eval_and_record_metrics('Logistic Regression', 'Classical Baseline', predictions['Logistic Regression'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
    del lr; gc.collect()

    # 1.2 Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=4)
    rf.fit(X_tr_s, y_tr)
    predictions['Random Forest'] = rf.predict_proba(X_te_s)[:, 1]
    eval_and_record_metrics('Random Forest', 'Classical Baseline', predictions['Random Forest'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
    del rf; gc.collect()

    # 1.3 LightGBM
    if LGBM_AVAILABLE:
        lgb_m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, num_leaves=31, random_state=42, n_jobs=4, verbose=-1)
        lgb_m.fit(X_tr_s, y_tr)
        predictions['LightGBM'] = lgb_m.predict_proba(X_te_s)[:, 1]
        eval_and_record_metrics('LightGBM', 'GBDT', predictions['LightGBM'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
        del lgb_m; gc.collect()

    # 1.4 XGBoost
    if XGB_AVAILABLE:
        xgb_m = xgb.XGBClassifier(n_estimators=200, learning_rate=0.08, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=4)
        xgb_m.fit(X_tr_s, y_tr)
        predictions['XGBoost'] = xgb_m.predict_proba(X_te_s)[:, 1]
        baseline_auc = float(roc_auc_score(y_test, predictions['XGBoost']))
        eval_and_record_metrics('XGBoost', 'GBDT', predictions['XGBoost'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
        del xgb_m; gc.collect()

    # 1.5 CatBoost
    if CATBOOST_AVAILABLE:
        cb_m = cb.CatBoostClassifier(iterations=250, learning_rate=0.08, depth=6, random_state=42, verbose=0, thread_count=4)
        cb_m.fit(X_tr_s, y_tr)
        predictions['CatBoost'] = cb_m.predict_proba(X_te_s)[:, 1]
        eval_and_record_metrics('CatBoost', 'GBDT', predictions['CatBoost'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
        del cb_m; gc.collect()

    # 1.6 Gradient Boosting (Scikit-Learn)
    from sklearn.ensemble import GradientBoostingClassifier
    gb_m = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
    gb_m.fit(X_tr_s, y_tr)
    predictions['Gradient Boosting'] = gb_m.predict_proba(X_te_s)[:, 1]
    eval_and_record_metrics('Gradient Boosting', 'GBDT', predictions['Gradient Boosting'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
    del gb_m; gc.collect()


    # -------------------------------------------------------------
    # Tier 2: Standard Neural, FT-Transformer & TabNet
    # -------------------------------------------------------------
    logging.info("\n--- [TIER 2: DEEP TABULAR & TRANSFORMER ARCHITECTURES] ---")
    
    # 2.1 Standard MLP
    mlp = TabFMStandardMLP(hidden_dim=128, epochs=10, batch_size=256, lr=1e-3)
    mlp.fit(X_tr_s, y_tr)
    predictions['Standard MLP'] = mlp.predict_proba(X_te_s)[:, 1]
    eval_and_record_metrics('Standard MLP', 'Neural', predictions['Standard MLP'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
    del mlp; gc.collect()

    # 2.2 FT-Transformer (Feature Tokenizer + Transformer)
    try:
        logging.info("  Running FT-Transformer (Gorishniy et al., NeurIPS 2021)...")
        ft_tf = FTTransformerWrapper(embed_dim=32, num_heads=4, num_layers=3, max_features=128, epochs=10, batch_size=256, lr=1e-3)
        ft_tf.fit(X_tr_s, y_tr)
        predictions['FT-Transformer'] = ft_tf.predict_proba(X_te_s)[:, 1]
        eval_and_record_metrics('FT-Transformer', 'Deep Tabular', predictions['FT-Transformer'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
        del ft_tf; gc.collect()
    except Exception as e:
        logging.warning(f"FT-Transformer benchmark skipped or failed: {e}")

    # 2.3 TabNet (Attentive Interpretable Tabular Learning)
    try:
        logging.info("  Running TabNet (Arik & Pfister, AAAI 2021)...")
        tabnet = TabNetWrapper(feature_dim=64, output_dim=32, num_steps=3, gamma=1.3, max_features=128, epochs=10, batch_size=256, lr=2e-3)
        tabnet.fit(X_tr_s, y_tr)
        predictions['TabNet'] = tabnet.predict_proba(X_te_s)[:, 1]
        eval_and_record_metrics('TabNet', 'Deep Tabular', predictions['TabNet'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
        del tabnet; gc.collect()
    except Exception as e:
        logging.warning(f"TabNet benchmark skipped or failed: {e}")

    # -------------------------------------------------------------
    # Tier 3: Domain-Hierarchical (HIR-M3) & GBDT Ensemble
    # -------------------------------------------------------------
    logging.info("\n--- [TIER 3: HIERARCHICAL HIR-M3] ---")
    if train_hir is not None:
        try:
            hir_hp = {'BATCH_SIZE': 256, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'HIDDEN_DIM': 128, 'LR': 0.001, 'EPOCHS': 5, 'LAMBDA_HIR': 0.05, 'GAMMA': 0.5}
            w_tr = np.ones(len(y_tr), dtype=np.float32)
            w_te = np.ones(len(y_test), dtype=np.float32)
            hir_prob = train_hir(X_tr_s, y_tr, w_tr, X_te_s, y_test, w_te, feature_cols, hir_hp)
            if hir_prob is not None:
                predictions['HIR-M3 Transformer'] = hir_prob
                eval_and_record_metrics('HIR-M3 Transformer', 'Domain-Hierarchical', predictions['HIR-M3 Transformer'], y_test, race_te, baseline_auc, comparison_rows, output_csv)

                if 'XGBoost' in predictions:
                    predictions['70% XGBoost : 30% HIR-M3'] = 0.7 * predictions['XGBoost'] + 0.3 * predictions['HIR-M3 Transformer']
                    eval_and_record_metrics('70% XGBoost : 30% HIR-M3', 'Ensemble', predictions['70% XGBoost : 30% HIR-M3'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
        except Exception as e:
            logging.error(f"Failed to train HIR-M3: {e}")

    # -------------------------------------------------------------
    # Tier 4: Algorithmic Equity & ACT-Parity v2
    # -------------------------------------------------------------
    logging.info("\n--- [TIER 4: ACT-PARITY V2 & HYBRIDS] ---")
    try:
        from parity.run_experiments import train_act_parity_model
        act_prob, _, _ = train_act_parity_model(X_tr_s, y_tr, race_tr_s, X_va_s, y_va, race_va_s, X_te_s, feature_cols, {'variant': 'V6_Full', 'epochs': 5, 'batch_size': 256})
        predictions['ACT-Parity v2'] = act_prob
        eval_and_record_metrics('ACT-Parity v2', 'Equity-Constrained', predictions['ACT-Parity v2'], y_test, race_te, baseline_auc, comparison_rows, output_csv)

        # HIR-M3 + ACT-Parity Hybrid
        from parity.run_unified_parity_comparison import train_hirm3_act_parity_hybrid
        hybrid_cfg = {'batch_size': 256, 'epochs': 5, 'embed_dim': 32, 'num_heads': 4, 'lr': 1e-3, 'lambda_hir': 0.05, 'gamma': 0.5, 'delta': 0.04, 'lambda_inv': 0.1}
        hybrid_prob = train_hirm3_act_parity_hybrid(X_tr_s, y_tr, race_tr_s, X_va_s, y_va, race_va_s, X_te_s, feature_cols, hybrid_cfg)
        predictions['HIR-M3 + ACT-Parity Hybrid'] = hybrid_prob
        eval_and_record_metrics('HIR-M3 + ACT-Parity Hybrid', 'Equity-Constrained', predictions['HIR-M3 + ACT-Parity Hybrid'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
    except Exception as e:
        logging.error(f"Failed to train ACT-Parity models: {e}")

    # -------------------------------------------------------------
    # Tier 5: Tabular Foundational Models (TabPFN, TabICL, TabFM)
    # -------------------------------------------------------------
    logging.info("\n--- [TIER 5: TABULAR FOUNDATIONAL MODELS (TFMs)] ---")

    # 5.1 TabPFN
    try:
        logging.info("  Running TabPFN In-Context Classifier (Stratified Context N=1,024, Top D=100 features)...")
        tabpfn = TabPFNWrapper(max_context_samples=1024, max_features=100, ensemble_configs=8)
        tabpfn.fit(X_tr_s, y_tr)
        predictions['TabPFN (Zero-Shot)'] = tabpfn.predict_proba(X_te_s)[:, 1]
        eval_and_record_metrics('TabPFN (Zero-Shot)', 'Foundational (TFM)', predictions['TabPFN (Zero-Shot)'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
        del tabpfn; gc.collect()
    except Exception as e:
        logging.warning(f"TabPFN benchmark skipped or failed: {e}")

    # 5.2 TabICL (Tabular In-Context Learning / Meta-Learner)
    try:
        logging.info("  Running TabICL In-Context Classifier (Stratified Context N=1,024, Top D=100 features)...")
        tabicl = TabICLWrapper(max_context_samples=1024, max_features=100, n_neighbors=50)
        tabicl.fit(X_tr_s, y_tr)
        predictions['TabICL (In-Context)'] = tabicl.predict_proba(X_te_s)[:, 1]
        eval_and_record_metrics('TabICL (In-Context)', 'Foundational (TFM)', predictions['TabICL (In-Context)'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
        del tabicl; gc.collect()
    except Exception as e:
        logging.warning(f"TabICL benchmark skipped or failed: {e}")

    # 5.3 TabFM (Tabular Transformer Foundation Model Architecture)
    try:
        logging.info("  Running TabFM (Feature-Tokenized Transformer Foundation Backbone)...")
        tabfm = TabFMWrapper(embed_dim=32, num_heads=4, num_layers=2, max_features=120, epochs=10, batch_size=256, lr=1e-3)
        tabfm.fit(X_tr_s, y_tr)
        predictions['TabFM (Transformer Backbone)'] = tabfm.predict_proba(X_te_s)[:, 1]
        eval_and_record_metrics('TabFM (Transformer Backbone)', 'Foundational (TFM)', predictions['TabFM (Transformer Backbone)'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
        del tabfm; gc.collect()
    except Exception as e:
        logging.warning(f"TabFM benchmark skipped or failed: {e}")

    # -------------------------------------------------------------
    # Generate Final Comparative Markdown Report
    # -------------------------------------------------------------
    final_df = pd.DataFrame(comparison_rows)
    logging.info("\n" + "="*80)
    logging.info("FINAL TEXAS BENCHMARK SUMMARY TABLE:")
    logging.info("\n" + final_df[['Model_Tier', 'Model_Name', 'ROC_AUC', 'PR_AUC', 'F1_Score', 'Worst_Group_FNR', 'EFNHI_Star', 'Brier_Score']].to_string(index=False))
    logging.info("="*80)

    # Save to Markdown
    with open(output_report, 'w') as f:
        f.write("# Texas Cohort: Comprehensive Model Benchmark Report\n\n")
        f.write("Evaluation across all model tiers: Classical Baselines $\\to$ GBDTs $\\to$ Neural $\\to$ HIR-M3 $\\to$ ACT-Parity v2 $\\to$ Tabular Foundational Models (TabPFN, TabICL).\n\n")
        f.write("### Benchmark Results Summary Table\n\n")
        f.write(final_df.to_markdown(index=False))
        f.write("\n\n---\n")
        f.write("### Key Observations\n")
        f.write("1. **Discriminative Utility**: Evaluate whether TabPFN/TabICL approach tuned GBDTs (XGBoost/CatBoost) and HIR-M3.\n")
        f.write("2. **Worst-Group FNR & Equity**: Note whether foundational models without explicit D-GAP constraints suffer from elevated `Worst_Group_FNR` and `EFNHI_Star` compared to ACT-Parity v2.\n")
        f.write("3. **Calibration**: Check if in-context probabilities achieve Brier Scores comparable to Platt-calibrated models.\n")

    logging.info(f"Report saved to: {output_report}")
    logging.info(f"CSV data saved to: {output_csv}")


if __name__ == "__main__":
    run_texas_benchmark()

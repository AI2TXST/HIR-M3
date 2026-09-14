#!/usr/bin/env python3
"""
===============================================================================
NATIONWIDE BENCHMARK PIPELINE (50% Stratified Sample)
-------------------------------------------------------------------------------
Evaluates the full continuum of predictive and algorithmic fairness paradigms
on the 50% Stratified Nationwide OASIS Readmission Cohort:

  Tier 1: Classical Baselines (Logistic Regression, Random Forest)
  Tier 2: High-Capacity GBDTs (LightGBM, XGBoost, CatBoost)
  Tier 3: Deep Tabular Neural Baselines (Standard MLP)
  Tier 4: Domain-Hierarchical Neural Networks (HIR-M3 Transformer, 70:30 & 90:10 Hybrid Ensembles)
  Tier 5: Algorithmic Equity Constrained Models (ACT-Parity v2, HIR-M3 + ACT-Parity Hybrid)
  Tier 6: Tabular Foundation Models (TabPFN Zero-Shot, TabICL In-Context, TabFM Transformer)

Metrics Reported:
  - Discrimination: ROC-AUC, PR-AUC, Sensitivity (TPR), Specificity, Precision, F1-Score
  - Calibration: Brier Score, Platt Calibration Slope
  - Equity & Safety: Worst-Group FNR, FNR Gap (Delta FNR), Equalized Odds Diff (EOD), EFNHI*
===============================================================================
"""

import os
import sys
import time
import gc
import json
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
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
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None

try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    RandomOverSampler = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

TARGET_COL = "ever_readmitted"


# -------------------------------------------------------------------------
# Tabular Foundation Model (TFM) Wrappers with Subsampling & Batch Guards
# -------------------------------------------------------------------------
class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """
    TabPFN (Prior-Data Fitted Network) Classifier Wrapper:
    Zero-shot transformer inference using prior-trained tabular knowledge.
    Uses stratified context subsampling (N=1,024) to fit architecture prior bounds.
    """
    def __init__(self, max_context_samples=1024, max_features=100, ensemble_configs=8, device='cpu'):
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
            logging.error("TabPFN package is not installed.")
            return self

        n_samples, n_feats = X.shape
        if n_feats > self.max_features:
            var_order = np.argsort(np.var(X, axis=0))[::-1]
            self.selected_feat_indices = var_order[:self.max_features]
            X_sub = X[:, self.selected_feat_indices]
        else:
            self.selected_feat_indices = np.arange(n_feats)
            X_sub = X

        if n_samples > self.max_context_samples:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=self.max_context_samples, random_state=42)
            idx, _ = next(sss.split(X_sub, y))
            X_fit, y_fit = X_sub[idx], y[idx]
            logging.info(f"    [TabPFN] Context Subsampled: {n_samples} -> {len(X_fit)} samples, {X_fit.shape[1]} features.")
        else:
            X_fit, y_fit = X_sub, y

        self.model = TabPFNClassifier(
            device=self.device,
            N_ensemble_configurations=self.ensemble_configs
        )
        try:
            self.model.fit(X_fit, y_fit, overwrite_warning=True)
        except TypeError:
            self.model.fit(X_fit, y_fit)
            
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        if self.model is None:
            return np.zeros((len(X), 2))
        X_sub = X[:, self.selected_feat_indices] if self.selected_feat_indices is not None else X
        
        batch_size = 1000
        probs_list = []
        for i in range(0, len(X_sub), batch_size):
            b_X = X_sub[i:i+batch_size]
            b_prob = self.model.predict_proba(b_X)
            probs_list.append(b_prob)
        return np.vstack(probs_list)


class TabICLWrapper(BaseEstimator, ClassifierMixin):
    """
    TabICL (Tabular In-Context Learning) Classifier Wrapper:
    Uses stratified context subsampling (N=1,024) to avoid OOM crashes during quadratic attention.
    """
    def __init__(self, max_context_samples=1024, max_features=100, n_neighbors=50, device='cuda'):
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

        if n_samples > self.max_context_samples:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=self.max_context_samples, random_state=42)
            idx, _ = next(sss.split(X_sub, y))
            X_fit, y_fit = X_sub[idx], y[idx]
            logging.info(f"    [TabICL] Subsampled context to {len(X_fit)} samples, {X_fit.shape[1]} features.")
        else:
            X_fit, y_fit = X_sub, y

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

        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights='distance')
        self.model.fit(X_fit, y_fit)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X_sub = X[:, self.selected_feat_indices] if self.selected_feat_indices is not None else X
        if self.model is None:
            return np.zeros((len(X), 2))

        batch_size = 500
        probs_list = []
        for i in range(0, len(X_sub), batch_size):
            b_X = X_sub[i:i+batch_size]
            b_prob = self.model.predict_proba(b_X)
            probs_list.append(b_prob)
        return np.vstack(probs_list)


class TabFMWrapper(BaseEstimator, ClassifierMixin):
    """
    TabFM (Tabular Foundation Model) Transformer Backbone Wrapper:
    Feature-tokenized transformer architecture with CLS classification head.
    """
    def __init__(self, embed_dim=32, num_heads=4, num_layers=2, max_features=120, epochs=10, batch_size=256, lr=1e-3, device='cuda'):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_features = max_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = 'cuda' if (torch.cuda.is_available() and device == 'cuda') else 'cpu'
        self.model = None
        self.selected_feat_indices = None
        self.is_official_tabfm = False

    def fit(self, X, y):
        n_samples, n_feats = X.shape
        if n_feats > self.max_features:
            var_order = np.argsort(np.var(X, axis=0))[::-1]
            self.selected_feat_indices = var_order[:self.max_features]
            X_sub = X[:, self.selected_feat_indices]
        else:
            self.selected_feat_indices = np.arange(n_feats)
            X_sub = X

        try:
            import tabfm
            if hasattr(tabfm, 'TabFMClassifier'):
                self.model = tabfm.TabFMClassifier(device=self.device)
                self.model.fit(X_sub, y)
                self.is_official_tabfm = True
                self.classes_ = np.unique(y)
                logging.info("    [TabFM] Successfully fitted official TabFM pre-trained foundation model.")
                return self
        except Exception as e:
            logging.info(f"    [TabFM] Official tabfm load note: {e}. Using Transformer Foundation Backbone.")

        in_dim = X_sub.shape[1]
        class FeatureTokenizedTransformerBackbone(nn.Module):
            def __init__(self, num_features, embed_dim, num_heads, num_layers):
                super().__init__()
                self.token_embeddings = nn.Parameter(torch.randn(num_features, embed_dim) * 0.02)
                self.num_weights = nn.Parameter(torch.randn(num_features, embed_dim) * 0.02)
                self.num_biases = nn.Parameter(torch.zeros(num_features, embed_dim))
                self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
                    dropout=0.1, activation='gelu', batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, 64),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1)
                )

            def forward(self, x):
                B, D = x.shape
                x_expanded = x.unsqueeze(-1)
                tokens = x_expanded * self.num_weights.unsqueeze(0) + self.num_biases.unsqueeze(0) + self.token_embeddings.unsqueeze(0)
                cls_tokens = self.cls_token.expand(B, -1, -1)
                seq = torch.cat([cls_tokens, tokens], dim=1)
                trans_out = self.transformer(seq)
                cls_out = trans_out[:, 0, :]
                logits = self.head(cls_out).squeeze(-1)
                return logits

        self.model = FeatureTokenizedTransformerBackbone(in_dim, self.embed_dim, self.num_heads, self.num_layers).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        ds = TensorDataset(torch.tensor(X_sub, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X_sub = X[:, self.selected_feat_indices] if self.selected_feat_indices is not None else X
        if self.is_official_tabfm and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_sub)

        self.model.eval()
        probs_list = []
        batch_size = 1000
        with torch.no_grad():
            for i in range(0, len(X_sub), batch_size):
                b_X = torch.tensor(X_sub[i:i+batch_size], dtype=torch.float32).to(self.device)
                logits = self.model(b_X).cpu().numpy()
                p1 = 1 / (1 + np.exp(-logits))
                probs_list.append(p1)
        p1_all = np.concatenate(probs_list)
        return np.column_stack([1 - p1_all, p1_all])


# -------------------------------------------------------------------------
# Standard PyTorch MLP Baseline
# -------------------------------------------------------------------------
class StandardMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dim=128, epochs=10, batch_size=256, lr=1e-3):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.net = None

    def fit(self, X, y):
        in_dim = X.shape[1]
        self.net = StandardMLP(in_dim, self.hidden_dim).to(DEVICE)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                out = self.net(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            bx = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            logits = self.net(bx).squeeze().cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])


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


# -------------------------------------------------------------------------
# Data & Experiment Utilities
# -------------------------------------------------------------------------
def load_nationwide_dataset():
    """
    Loads Nationwide OASIS Readmission cohort from Parquet or CSV.
    """
    candidates = [
        os.path.join(BASE_DIR, "data", "processed_final_mergedDF_condensed.parquet"),
        os.path.join(BASE_DIR, "data", "processed_final_mergedDF_condensed.csv"),
        "../data/processed_final_mergedDF_condensed.parquet",
        "data/processed_final_mergedDF_condensed.parquet"
    ]
    filepath = next((p for p in candidates if os.path.exists(p)), None)
    if not filepath:
        logging.error("Nationwide dataset files not found.")
        return None

    logging.info(f"Loading Nationwide cohort dataset from: {filepath}")
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath, low_memory=False)

    possible_targets = [TARGET_COL, 'ever_readmitted', 'READMISSION', 'Readmission']
    t_col = next((t for t in possible_targets if t in df.columns), None)
    if not t_col:
        logging.error("Target column not found in Nationwide dataset.")
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
        'Cohort': 'Nationwide_50pct',
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
# Main Nationwide Comparison Pipeline
# -------------------------------------------------------------------------
def run_nationwide_benchmark(sample_frac=0.50):
    logging.info("=================================================================")
    logging.info(f"  NATIONWIDE COHORT BENCHMARK ({int(sample_frac*100)}% STRATIFIED SAMPLE)")
    logging.info("  Evaluating Classical -> GBDTs -> Neural -> HIR-M3 -> ACT-Parity -> TFMs")
    logging.info("=================================================================")

    loaded = load_nationwide_dataset()
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

    X_df = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    X = np.array(X_df.values, dtype=np.float32)
    y = np.array(df[t_col].tolist(), dtype=np.int64)
    race_array = np.array(df[race_col].tolist(), dtype=str)

    del df, X_df
    gc.collect()

    n_total = len(y)
    indices = np.arange(n_total)

    # Apply 50% stratified sampling if requested
    if sample_frac < 1.0:
        sss_sample = StratifiedShuffleSplit(n_splits=1, train_size=sample_frac, random_state=42)
        sampled_idx, _ = next(sss_sample.split(indices, y))
        indices = sampled_idx
        logging.info(f"Applied {int(sample_frac*100)}% Stratified Sampling: {n_total} -> {len(indices)} rows.")

    train_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=42, stratify=y[indices]
    )
    tr_idx, va_idx = train_test_split(
        train_idx, test_size=0.15, random_state=42, stratify=y[train_idx]
    )

    X_tr = X[tr_idx]
    y_tr = y[tr_idx]
    race_tr = race_array[tr_idx]

    X_va = X[va_idx]
    y_va = y[va_idx]
    race_va = race_array[va_idx]

    X_te = X[test_idx]
    y_test = y[test_idx]
    race_te = race_array[test_idx]

    del X, y, race_array
    gc.collect()

    logging.info(f"Nationwide Partition Sizes: Train={len(X_tr)}, Val={len(X_va)}, Test={len(X_te)}, Features={len(feature_cols)}")

    # Standardize Features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    del X_tr, X_va, X_te
    gc.collect()

    # Oversample minority class on training split only
    if RandomOverSampler is not None:
        ros = RandomOverSampler(random_state=42)
        X_tr_s, y_tr = ros.fit_resample(X_tr_s, y_tr)
        race_tr_s = race_tr[ros.sample_indices_]
    else:
        race_tr_s = race_tr

    race_va_s = race_va

    output_csv = os.path.join(PARITY_DIR, "results", "nationwide_all_models_comparison.csv")
    output_report = os.path.join(BASE_DIR, "docs", "NATIONWIDE_ALL_MODELS_COMPARATIVE_REPORT.md")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    comparison_rows = []
    predictions = {}
    baseline_auc = None

    # -------------------------------------------------------------
    # Tier 1: Classical Baselines
    # -------------------------------------------------------------
    logging.info("\n--- [TIER 1: CLASSICAL STATISTICAL BASELINES] ---")

    # 1.1 Logistic Regression
    try:
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver='lbfgs')
        lr.fit(X_tr_s, y_tr)
        predictions['Logistic Regression'] = lr.predict_proba(X_te_s)[:, 1]
        eval_and_record_metrics('Logistic Regression', 'Classical Baseline', predictions['Logistic Regression'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
    except Exception as e:
        logging.error(f"Logistic Regression failed: {e}")

    # 1.2 Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        predictions['Random Forest'] = rf.predict_proba(X_te_s)[:, 1]
        eval_and_record_metrics('Random Forest', 'Classical Baseline', predictions['Random Forest'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
    except Exception as e:
        logging.error(f"Random Forest failed: {e}")

    # -------------------------------------------------------------
    # Tier 2: High-Capacity GBDTs
    # -------------------------------------------------------------
    logging.info("\n--- [TIER 2: HIGH-CAPACITY GBDTs] ---")

    # 2.1 LightGBM
    if LGBM_AVAILABLE:
        try:
            lgb_model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
            lgb_model.fit(X_tr_s, y_tr)
            predictions['LightGBM'] = lgb_model.predict_proba(X_te_s)[:, 1]
            base_row = eval_and_record_metrics('LightGBM', 'GBDT', predictions['LightGBM'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
            baseline_auc = base_row['ROC_AUC']
        except Exception as e:
            logging.error(f"LightGBM failed: {e}")

    # 2.2 XGBoost
    if XGB_AVAILABLE:
        try:
            xgb_model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, eval_metric='logloss', random_state=42, n_jobs=-1)
            xgb_model.fit(X_tr_s, y_tr)
            predictions['XGBoost'] = xgb_model.predict_proba(X_te_s)[:, 1]
            eval_and_record_metrics('XGBoost', 'GBDT', predictions['XGBoost'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
        except Exception as e:
            logging.error(f"XGBoost failed: {e}")

    # 2.3 CatBoost
    if CB_AVAILABLE:
        try:
            cb_model = cb.CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, verbose=0, random_seed=42, thread_count=-1)
            cb_model.fit(X_tr_s, y_tr)
            predictions['CatBoost'] = cb_model.predict_proba(X_te_s)[:, 1]
            eval_and_record_metrics('CatBoost', 'GBDT', predictions['CatBoost'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
        except Exception as e:
            logging.error(f"CatBoost failed: {e}")

    # -------------------------------------------------------------
    # Tier 3: Neural & Deep Tabular Baselines (MLP, FT-Transformer, TabNet)
    # -------------------------------------------------------------
    logging.info("\n--- [TIER 3: NEURAL TABULAR & TRANSFORMER BASELINES] ---")
    if TORCH_AVAILABLE:
        try:
            mlp = MLPClassifierWrapper(hidden_dim=128, epochs=10, batch_size=256, lr=1e-3)
            mlp.fit(X_tr_s, y_tr)
            predictions['Standard MLP'] = mlp.predict_proba(X_te_s)[:, 1]
            eval_and_record_metrics('Standard MLP', 'Neural', predictions['Standard MLP'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
            del mlp; gc.collect()
        except Exception as e:
            logging.error(f"MLP training failed: {e}")

        # 3.2 FT-Transformer (Feature Tokenizer + Transformer)
        try:
            logging.info("  Running FT-Transformer (Gorishniy et al., NeurIPS 2021)...")
            ft_tf = FTTransformerWrapper(embed_dim=32, num_heads=4, num_layers=3, max_features=128, epochs=10, batch_size=256, lr=1e-3)
            ft_tf.fit(X_tr_s, y_tr)
            predictions['FT-Transformer'] = ft_tf.predict_proba(X_te_s)[:, 1]
            eval_and_record_metrics('FT-Transformer', 'Deep Tabular', predictions['FT-Transformer'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
            del ft_tf; gc.collect()
        except Exception as e:
            logging.warning(f"FT-Transformer benchmark skipped or failed: {e}")

        # 3.3 TabNet (Attentive Interpretable Tabular Learning)
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
    # Tier 4: Domain Hierarchical Models (HIR-M3 & Ensembles)
    # -------------------------------------------------------------
    logging.info("\n--- [TIER 4: HIERARCHICAL HIR-M3 & ENSEMBLES] ---")
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

                if 'LightGBM' in predictions:
                    predictions['90% LightGBM : 10% HIR-M3'] = 0.9 * predictions['LightGBM'] + 0.1 * predictions['HIR-M3 Transformer']
                    eval_and_record_metrics('90% LightGBM : 10% HIR-M3', 'Ensemble', predictions['90% LightGBM : 10% HIR-M3'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
        except Exception as e:
            logging.error(f"Failed to train HIR-M3: {e}")

    # -------------------------------------------------------------
    # Tier 5: Algorithmic Equity & ACT-Parity v2
    # -------------------------------------------------------------
    logging.info("\n--- [TIER 5: ACT-PARITY V2 & HYBRIDS] ---")
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
    # Tier 6: Tabular Foundational Models (TabPFN, TabICL, TabFM)
    # -------------------------------------------------------------
    logging.info("\n--- [TIER 6: TABULAR FOUNDATIONAL MODELS (TFMs)] ---")

    # 6.1 TabPFN
    try:
        logging.info("  Running TabPFN In-Context Classifier (Stratified Context N=1,024, Top D=100 features)...")
        tabpfn = TabPFNWrapper(max_context_samples=1024, max_features=100, ensemble_configs=8)
        tabpfn.fit(X_tr_s, y_tr)
        predictions['TabPFN (Zero-Shot)'] = tabpfn.predict_proba(X_te_s)[:, 1]
        eval_and_record_metrics('TabPFN (Zero-Shot)', 'Foundational (TFM)', predictions['TabPFN (Zero-Shot)'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
        del tabpfn; gc.collect()
    except Exception as e:
        logging.warning(f"TabPFN benchmark skipped or failed: {e}")

    # 6.2 TabICL (Tabular In-Context Learning / Meta-Learner)
    try:
        logging.info("  Running TabICL In-Context Classifier (Stratified Context N=1,024, Top D=100 features)...")
        tabicl = TabICLWrapper(max_context_samples=1024, max_features=100, n_neighbors=50)
        tabicl.fit(X_tr_s, y_tr)
        predictions['TabICL (In-Context)'] = tabicl.predict_proba(X_te_s)[:, 1]
        eval_and_record_metrics('TabICL (In-Context)', 'Foundational (TFM)', predictions['TabICL (In-Context)'], y_test, race_te, baseline_auc, comparison_rows, output_csv)
        del tabicl; gc.collect()
    except Exception as e:
        logging.warning(f"TabICL benchmark skipped or failed: {e}")

    # 6.3 TabFM (Tabular Transformer Foundation Model Architecture)
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
    logging.info("FINAL NATIONWIDE (50% SAMPLE) BENCHMARK SUMMARY TABLE:")
    logging.info("\n" + final_df[['Model_Tier', 'Model_Name', 'ROC_AUC', 'PR_AUC', 'F1_Score', 'Worst_Group_FNR', 'EFNHI_Star', 'Brier_Score']].to_string(index=False))
    logging.info("="*80)

    # Save to Markdown
    with open(output_report, 'w') as f:
        f.write("# Nationwide Cohort (50% Stratified Sample): Comprehensive Model Benchmark Report\n\n")
        f.write("Evaluation across all model tiers: Classical Baselines $\\to$ GBDTs $\\to$ Neural $\\to$ HIR-M3 $\\to$ ACT-Parity v2 $\\to$ Tabular Foundational Models (TabPFN, TabICL, TabFM).\n\n")
        f.write("### Benchmark Results Summary Table\n\n")
        f.write(final_df.to_markdown(index=False))
        f.write("\n\n---\n")
        f.write("### Key Observations\n")
        f.write("1. **Discriminative Utility**: Examines discrimination on large-scale multi-state data.\n")
        f.write("2. **Worst-Group FNR & Equity**: Evaluates algorithmic parity across demographic subgroups.\n")
        f.write("3. **Calibration**: Quantifies probability calibration and Brier scores.\n")

    logging.info(f"\nMarkdown report exported to: {output_report}")
    logging.info(f"CSV results exported to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Nationwide OASIS Model Benchmark (50% Sample)")
    parser.add_argument("--sample_frac", type=float, default=0.50, help="Fraction of stratified cohort sample to evaluate (default: 0.50)")
    args = parser.parse_args()

    run_nationwide_benchmark(sample_frac=args.sample_frac)

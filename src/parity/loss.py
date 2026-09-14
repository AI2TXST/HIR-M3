import logging
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if TORCH_AVAILABLE:
    class DualMultiplierManager:
        """
        Tracks and updates epoch-level Augmented Lagrangian dual multipliers (lambda_g)
        over validation / full dataset evaluation, preventing noisy per-batch updates.
        """
        def __init__(self, initial_lambda=0.0, eta=0.05, max_lambda=10.0):
            self.eta = eta
            self.max_lambda = max_lambda
            self.lambdas = {}

        def get_lambda(self, group_id):
            return self.lambdas.get(group_id, 0.0)

        def get_all_lambdas(self):
            return self.lambdas

        def update(self, constraint_dict):
            """
            Epoch update rule: lambda_g <- max(0, lambda_g + eta * c_g)
            """
            for g, c_g in constraint_dict.items():
                current = self.lambdas.get(g, 0.0)
                updated = max(0.0, current + self.eta * c_g)
                self.lambdas[g] = min(updated, self.max_lambda)

    class AugmentedLagrangianDGAPLoss(nn.Module):
        """
        Augmented Lagrangian Dynamic Group Adaptive Parity (D-GAP) Loss.
        Constrains soft differentiable subgroup False Negative Rate (FNR) relative to 
        overall population FNR + clinical tolerance delta.
        Includes tier invariance regularization (L_inv) and support thresholding.
        """
        def __init__(self, delta=0.04, rho=1.0, min_support=30, lambda_inv=0.1, pos_weight=None):
            super().__init__()
            self.delta = delta
            self.rho = rho
            self.min_support = min_support
            self.lambda_inv = lambda_inv
            if pos_weight is not None:
                if not isinstance(pos_weight, torch.Tensor):
                    pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
                self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
            else:
                self.bce = nn.BCEWithLogitsLoss(reduction='none')

        def forward(self, logits, targets, group_ids, lambda_dict, logits_perturbed=None):
            """
            Computes Total Augmented Lagrangian Loss:
            L = L_BCE + sum_g [ lambda_g * c_g + (rho / 2) * max(0, c_g)^2 ] + lambda_inv * L_inv
            """
            probs = torch.sigmoid(logits).squeeze()
            targets = targets.squeeze()
            
            # Primary Binary Cross-Entropy Loss
            bce_loss = self.bce(logits.squeeze(), targets).mean()
            
            # Soft Population FNR
            pos_mask_all = (targets == 1)
            if pos_mask_all.sum() == 0:
                return bce_loss, {}

            fnr_all = 1.0 - (probs[pos_mask_all].sum() / (pos_mask_all.sum() + 1e-8))
            
            penalty_loss = torch.tensor(0.0, device=logits.device)
            constraint_values = {}

            unique_groups = torch.unique(group_ids)
            for g in unique_groups:
                g_key = str(g.item())
                group_mask = (group_ids == g)
                pos_mask_g = group_mask & pos_mask_all
                
                n_pos_g = pos_mask_g.sum().item()
                if n_pos_g < self.min_support:
                    continue  # Support thresholding: skip sparse strata to prevent noisy updates

                # Differentiable Soft Subgroup FNR
                fnr_g = 1.0 - (probs[pos_mask_g].sum() / (n_pos_g + 1e-8))
                
                # Clinical constraint: c_g = FNR_g - (FNR_all + delta) <= 0
                c_g = fnr_g - (fnr_all + self.delta)
                constraint_values[g_key] = c_g.item()

                lam = lambda_dict.get(g_key, 0.0)
                c_g_pos = torch.clamp(c_g, min=0.0)
                
                # Augmented Lagrangian Penalty: lam * c_g + (rho / 2) * max(0, c_g)^2
                penalty_loss = penalty_loss + lam * c_g + (self.rho / 2.0) * (c_g_pos ** 2)

            # Tier Invariance Regularization (L_inv)
            inv_loss = torch.tensor(0.0, device=logits.device)
            if logits_perturbed is not None:
                probs_perturbed = torch.sigmoid(logits_perturbed).squeeze()
                inv_loss = F.mse_loss(probs, probs_perturbed)

            total_loss = bce_loss + penalty_loss + self.lambda_inv * inv_loss
            return total_loss, constraint_values

    class HIRM3_ACTParity_HybridLoss(nn.Module):
        """
        Unified Hybrid Loss Function combining:
        1. HIR-M3 Cross-Tier Penalty: L_HIR = Intra_Meso - gamma * Cross_Meso_Micro
        2. Augmented Lagrangian D-GAP Parity Loss: L_D-GAP = sum_g [ lambda_g * c_g + (rho/2) * max(0, c_g)^2 ]
        3. Tier Invariance Loss: L_inv
        """
        def __init__(self, delta=0.04, rho=1.0, min_support=30, lambda_hir=0.1, gamma=0.5, lambda_inv=0.1):
            super().__init__()
            self.dgap_loss = AugmentedLagrangianDGAPLoss(delta=delta, rho=rho, min_support=min_support, lambda_inv=lambda_inv)
            self.lambda_hir = lambda_hir
            self.gamma = gamma

        def forward(self, logits, targets, group_ids, lambda_dict, attn_weights=None, micro_idxs=None, meso_idxs=None, logits_perturbed=None):
            # Compute primary D-GAP Loss (BCE + Parity Penalty + Invariance)
            dgap_tot, c_dict = self.dgap_loss(logits, targets, group_ids, lambda_dict, logits_perturbed=logits_perturbed)

            # Compute HIR-M3 Cross-Tier Penalty
            hir_penalty = torch.tensor(0.0, device=logits.device)
            if attn_weights is not None and micro_idxs is not None and meso_idxs is not None and len(micro_idxs) > 0 and len(meso_idxs) > 0:
                if not isinstance(meso_idxs, torch.Tensor):
                    meso_t = torch.tensor(meso_idxs, dtype=torch.long, device=logits.device)
                else:
                    meso_t = meso_idxs.to(logits.device)
                if not isinstance(micro_idxs, torch.Tensor):
                    micro_t = torch.tensor(micro_idxs, dtype=torch.long, device=logits.device)
                else:
                    micro_t = micro_idxs.to(logits.device)

                try:
                    attn_meso = attn_weights.index_select(1, meso_t)
                    intra_meso = attn_meso.index_select(2, meso_t).mean()
                    cross_meso_micro = attn_meso.index_select(2, micro_t).mean()
                    hir_penalty = intra_meso - self.gamma * cross_meso_micro
                except Exception:
                    hir_penalty = torch.tensor(0.0, device=logits.device)

            total_hybrid_loss = dgap_tot + self.lambda_hir * hir_penalty
            return total_hybrid_loss, c_dict


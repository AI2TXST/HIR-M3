import os
import sys
import logging
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
DEVICE = torch.device("cpu") if TORCH_AVAILABLE else None

def get_m3_feature_groups_tokenized(all_features, remove_memorization_ids=True):
    """
    Categorizes column names into Micro (Clinical), Meso (SDOH/Geography), 
    and Macro (System/Care Process) token feature sets.
    Optionally strips raw agency/facility ID flags to prevent memorization shortcuts.
    """
    all_features_str = [str(f) for f in all_features]
    feats = set(all_features_str)

    def valid(names):
        return [f for f in names if f in feats]

    raw_id_cols = [c for c in all_features_str if c.startswith(('Agency_Medicare_Number', 'Facility_Internal_ID'))]
    
    # Micro Tier: Individual clinical utilization, comorbidities, ICD-10 hierarchy, demographics
    micro = valid([
        'Age', 'American_Indian_or_Alaska_Native', 'Asian', 'Black_or_African_American',
        'Hispanic_or_Latino', 'Native_Hawiian_or_Pacific_Islander', 'White',
        'BMI_Category_Obese-Class1', 'BMI_Category_Obese-Class2', 
        'BMI_Category_Obese-Class3', 'BMI_Category_Overweight', 'BMI_Category_Underweight',
        'ByDiscipline_PT', 'ByDiscipline_RN', 'ByDiscipline_SLP/ST',
        'charlson_score', 'charlson_ageadj', 'elix_quan_score', 'aids', 'alcohol', 'ami', 
        'canc', 'carit', 'cevd', 'chf', 'coag', 'copd', 'dane', 'dementia', 'depre', 'drug', 'fed',
        'hp', 'hypc', 'hypothy', 'hypunc', 'ld', 'lymph', 'metacanc', 'obes', 'ond', 'pcd', 'psycho',
        'pvd', 'rend', 'rheumd', 'valv', 'wloss'
    ] + [
        c for c in all_features_str 
        if c.startswith(('Primary_', 'Other1_', 'Other2_', 'Other3_', 'Other4_', 'Other5_', 'hicd_bert_emb_', 'icd_g', 'Primary_Diagnosis_', 'Other_Diagnosis_'))
    ])

    # Meso Tier: Neighborhood SDOH, rurality, census tracts, educational attainment, poverty
    meso = valid([
        c for c in all_features_str if c.startswith('COUNTY_NAME')
    ] + [
        'POP_URB', 'POPPCT_URB', 'POP_RUR', 'POPPCT_RUR', 'RUCA_Category',
        'ACS_PCT_BACHELOR_DGR', 'ACS_PCT_COLLEGE_ASSOCIATE_DGR', 'ACS_PCT_LT_HS',
        'ACS_PCT_NO_WORK_NO_SCHL_16_19', 'ACS_PCT_VET_COLLEGE',
        'ACS_PCT_HH_LIMIT_ENGLISH', 'ACS_PCT_HH_BROADBAND_ONLY',
        'ACS_PCT_HH_CELLULAR_ONLY', 'ACS_PCT_HH_DIAL_INTERNET_ONLY',
        'ACS_PCT_HH_INTERNET_NO_SUBS', 'ACS_PCT_HH_OTHER_COMP',
        'ACS_PCT_HH_OTHER_COMP_ONLY', 'ACS_PCT_HH_PC_ONLY',
        'ACS_PCT_HH_SAT_INTERNET', 'ACS_PCT_HH_TABLET_ONLY',
        'ACS_PCT_CHILDREN_GRANDPARENT', 'ACS_PCT_CHILD_1FAM',
        'ACS_PCT_GRANDP_NO_RESPS', 'ACS_PCT_GRANDP_RESPS_NO_P', 'ACS_PCT_GRANDP_RESPS_P',
        'ACS_PCT_HH_1PERS', 'ACS_PCT_HH_ABOVE65', 'ACS_TOT_GRANDCHILDREN_GP',
        'ACS_PCT_HEALTH_INC_138_199', 'ACS_PCT_HEALTH_INC_200_399',
        'ACS_PCT_HH_NO_FD_STMP_BLW_POV', 'ACS_PCT_INC50_ABOVE65',
        'ACS_PCT_POV_AIAN', 'ACS_PCT_POV_ASIAN', 'ACS_PCT_POV_BLACK', 'ACS_PCT_POV_HISPANIC',
        'ACS_PCT_POV_MULTI', 'ACS_PCT_POV_NHPI', 'ACS_PCT_POV_OTHER',
        'ACS_PCT_VET_POV_18_64', 'ACS_TOT_POP_POV'
    ])

    # Macro Tier: System/care process features, aggregated agency quality
    macro = [c for c in all_features_str if c not in set(micro + meso + (raw_id_cols if remove_memorization_ids else []))]

    if remove_memorization_ids:
        micro = [c for c in micro if c not in raw_id_cols]
        meso = [c for c in meso if c not in raw_id_cols]
        macro = [c for c in macro if c not in raw_id_cols]

    return micro, meso, macro

if TORCH_AVAILABLE:
    class ACTParityV2(nn.Module):
        """
        Tokenized ACT-Parity v2 Architecture.
        Maps Micro (Clinical), Meso (SDOH), and Macro (System) features into separate token 
        embeddings, applies Query-Key-Value Cross-Attention, and enforces residual clinical anchoring.
        """
        def __init__(self, num_micro, num_meso, num_macro, embed_dim=32, num_heads=4, alpha=0.5, dropout=0.1):
            super().__init__()
            self.num_micro = num_micro
            self.num_meso = num_meso
            self.num_macro = num_macro
            self.alpha = alpha

            # 1D Feature Token Projections
            self.micro_proj = nn.Linear(1, embed_dim)
            self.meso_proj = nn.Linear(1, embed_dim) if num_meso > 0 else None
            self.macro_proj = nn.Linear(1, embed_dim) if num_macro > 0 else None

            # Multi-Head Cross-Attention: Query = Micro Tokens, Keys/Values = Context (Meso + Macro)
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
            self.norm_attn = nn.LayerNorm(embed_dim)

            # Contextual Gate Network
            self.gate_net = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )

            # Final Classification Head
            self.head = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )

        def forward(self, x_micro, x_meso=None, x_macro=None):
            B = x_micro.size(0)
            
            # Project features to 3D Token Tensors (Batch, Num_Tokens, Embed_Dim)
            tokens_micro = self.micro_proj(x_micro.unsqueeze(-1))
            
            context_tokens = []
            if x_meso is not None and self.meso_proj is not None and x_meso.size(1) > 0:
                context_tokens.append(self.meso_proj(x_meso.unsqueeze(-1)))
            if x_macro is not None and self.macro_proj is not None and x_macro.size(1) > 0:
                context_tokens.append(self.macro_proj(x_macro.unsqueeze(-1)))

            if context_tokens:
                tokens_context = torch.cat(context_tokens, dim=1)
                attn_out, attn_weights = self.cross_attn(
                    query=tokens_micro, 
                    key=tokens_context, 
                    value=tokens_context
                )
                tokens_fused = self.norm_attn(tokens_micro + attn_out)
            else:
                tokens_fused = tokens_micro
                attn_weights = None

            # Mean-pool over sequence token dimensions
            h_micro = tokens_micro.mean(dim=1)
            h_context = tokens_fused.mean(dim=1)

            # Calculate Context Gate: g = Sigmoid(Linear([h_micro || h_context]))
            g = self.gate_net(torch.cat([h_micro, h_context], dim=-1))

            # Residual Clinical Anchoring: h_final = h_micro + alpha * (g * h_context)
            h_final = h_micro + self.alpha * (g * h_context)
            logits = self.head(h_final)

            return logits, g, attn_weights

    class HIRM3_ACTParity_Hybrid(nn.Module):
        """
        Unified Hybrid Architecture combining HIR-M3 Multi-Tier Feature Self-Attention
        with ACT-Parity v2 Tokenized QKV Cross-Attention and Residual Clinical Anchoring.
        """
        def __init__(self, num_micro, num_meso, num_macro, embed_dim=32, num_heads=4, alpha=0.5, dropout=0.1):
            super().__init__()
            self.act_parity = ACTParityV2(
                num_micro=num_micro, num_meso=num_meso, num_macro=num_macro,
                embed_dim=embed_dim, num_heads=num_heads, alpha=alpha, dropout=dropout
            )

        def forward(self, x_micro, x_meso=None, x_macro=None):
            logits, g, attn_weights = self.act_parity(x_micro, x_meso, x_macro)
            return logits, g, attn_weights


import re
import torch
import torch.nn as nn
import torch.nn.functional as F

def parse_icd_hierarchy(code: str):
    """
    Parses an ICD-10 code string into three hierarchy levels:
      - G0: 1-letter chapter prefix (e.g., 'I' for 'I10', 'E' for 'E11.9')
      - G1: 1-2 char prefix ignoring punctuation (e.g., 'I1' for 'I10', 'E1' for 'E11.9')
      - G2: 3-char category (e.g., 'I10' for 'I10', 'E11' for 'E11.9')
      
    Returns: tuple (g0, g1, g2)
    """
    if not code or not isinstance(code, str):
        return ("<PAD>", "<PAD>", "<PAD>")
    
    clean = re.sub(r'[^a-zA-Z0-9]', '', code.strip().upper())
    if not clean or clean in ("PAD", "<PAD>", "NONE", "NAN", "NULL"):
        return ("<PAD>", "<PAD>", "<PAD>")
    
    g0 = clean[:1] if len(clean) >= 1 else "<PAD>"
    g1 = clean[:2] if len(clean) >= 2 else g0
    g2 = clean[:3] if len(clean) >= 3 else g1
    
    return (g0, g1, g2)


class HierICDEmbedding(nn.Module):
    """
    Hierarchy-Aware ICD Embedding Layer for PyTorch.
    Computes additive embeddings across 4 levels:
        e(c) = e_code(c) + e_G0(G0(c)) + e_G1(G1(c)) + e_G2(G2(c))
    """
    def __init__(self, code_list, embed_dim=32, pad_token="<PAD>"):
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_token = pad_token
        
        # 1. Build Code Vocabulary (padding index = 0)
        unique_codes = sorted(list(set([
            c for c in code_list 
            if c and str(c).upper() not in ('NAN', 'NONE', '<PAD>', 'NULL')
        ])))
        self.code_to_id = {pad_token: 0}
        for idx, code in enumerate(unique_codes, start=1):
            self.code_to_id[code] = idx
        self.id_to_code = {v: k for k, v in self.code_to_id.items()}
        
        # 2. Build G0, G1, G2 Vocabularies
        g0_set, g1_set, g2_set = set(), set(), set()
        code_hierarchy_map = {}
        for code, code_id in self.code_to_id.items():
            g0, g1, g2 = parse_icd_hierarchy(code)
            code_hierarchy_map[code_id] = (g0, g1, g2)
            if g0 != pad_token: g0_set.add(g0)
            if g1 != pad_token: g1_set.add(g1)
            if g2 != pad_token: g2_set.add(g2)
            
        self.g0_to_id = {pad_token: 0}
        for idx, g in enumerate(sorted(list(g0_set)), start=1):
            self.g0_to_id[g] = idx
            
        self.g1_to_id = {pad_token: 0}
        for idx, g in enumerate(sorted(list(g1_set)), start=1):
            self.g1_to_id[g] = idx
            
        self.g2_to_id = {pad_token: 0}
        for idx, g in enumerate(sorted(list(g2_set)), start=1):
            self.g2_to_id[g] = idx

        # 3. Create pre-mapped ID index tensors (registered as buffers)
        num_codes = len(self.code_to_id)
        c2g0 = torch.zeros(num_codes, dtype=torch.long)
        c2g1 = torch.zeros(num_codes, dtype=torch.long)
        c2g2 = torch.zeros(num_codes, dtype=torch.long)
        
        for code_id, (g0, g1, g2) in code_hierarchy_map.items():
            c2g0[code_id] = self.g0_to_id.get(g0, 0)
            c2g1[code_id] = self.g1_to_id.get(g1, 0)
            c2g2[code_id] = self.g2_to_id.get(g2, 0)
            
        self.register_buffer("code_to_g0", c2g0)
        self.register_buffer("code_to_g1", c2g1)
        self.register_buffer("code_to_g2", c2g2)
        
        # 4. Trainable Embedding Tables
        self.code_emb = nn.Embedding(num_codes, embed_dim, padding_idx=0)
        self.g0_emb = nn.Embedding(len(self.g0_to_id), embed_dim, padding_idx=0)
        self.g1_emb = nn.Embedding(len(self.g1_to_id), embed_dim, padding_idx=0)
        self.g2_emb = nn.Embedding(len(self.g2_to_id), embed_dim, padding_idx=0)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Random initialization with zero-vector padding."""
        for emb in [self.code_emb, self.g0_emb, self.g1_emb, self.g2_emb]:
            nn.init.normal_(emb.weight, std=0.02)
            with torch.no_grad():
                emb.weight[0].zero_()

    def forward(self, code_ids):
        """
        code_ids: Tensor of shape (batch_size, num_codes_per_episode)
        Returns: 
            e(c) = e_code(c) + e_G0(G0(c)) + e_G1(G1(c)) + e_G2(G2(c))
            Shape: (batch_size, num_codes_per_episode, embed_dim)
        """
        g0_ids = self.code_to_g0[code_ids]
        g1_ids = self.code_to_g1[code_ids]
        g2_ids = self.code_to_g2[code_ids]
        
        e_code = self.code_emb(code_ids)
        e_g0 = self.g0_emb(g0_ids)
        e_g1 = self.g1_emb(g1_ids)
        e_g2 = self.g2_emb(g2_ids)
        
        return e_code + e_g0 + e_g1 + e_g2

    def pool_episode_embeddings(self, code_embs, code_ids, method="mean"):
        """
        Aggregates code embeddings for each patient episode into a fixed-size vector.
        code_embs: (batch_size, num_codes_per_episode, embed_dim)
        code_ids: (batch_size, num_codes_per_episode)
        Returns: (batch_size, embed_dim)
        """
        mask = (code_ids != 0).unsqueeze(-1).float()  # (batch_size, num_codes, 1)
        if method == "mean":
            sum_embs = (code_embs * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return sum_embs / counts
        else:
            raise ValueError(f"Unsupported pooling method: {method}")

    def encode_code_list(self, batch_code_strings):
        """Helper to convert nested list of ICD strings into a padded PyTorch LongTensor."""
        batch_size = len(batch_code_strings)
        max_len = max([len(codes) for codes in batch_code_strings]) if batch_code_strings else 0
        tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, codes in enumerate(batch_code_strings):
            for j, code in enumerate(codes):
                tensor[i, j] = self.code_to_id.get(code, 0)
        return tensor


# ----------------------------------------------------------------------
# Minimal Integration Stub for HIR-M3 Model
# ----------------------------------------------------------------------
class HIR_M3_ModelStub(nn.Module):
    """
    Integration stub illustrating how to combine HierICDEmbedding with 
    tabular micro features inside the HIR-M3 model architecture.
    """
    def __init__(self, num_tabular_features, code_list, embed_dim=32, hidden_dim=64):
        super().__init__()
        # Tabular Feature Projections
        self.tabular_linear = nn.Linear(num_tabular_features, embed_dim)
        
        # Hierarchical ICD Embedding Layer
        self.icd_layer = HierICDEmbedding(code_list, embed_dim=embed_dim)
        
        # Downstream Dense Classifier Head
        # Concatenated micro tier features: tabular_proj + icd_micro_features
        self.head = nn.Sequential(
            nn.Linear(embed_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_tabular, x_icd_codes):
        """
        x_tabular: (batch_size, num_tabular_features)
        x_icd_codes: (batch_size, num_codes_per_episode)
        """
        # 1. Tabular features projection
        tab_proj = self.tabular_linear(x_tabular)  # (batch_size, embed_dim)
        
        # 2. Hierarchical ICD embedding lookup & episode aggregation
        code_embs = self.icd_layer(x_icd_codes)   # (batch_size, num_codes, embed_dim)
        icd_micro_features = self.icd_layer.pool_episode_embeddings(code_embs, x_icd_codes) # (batch_size, embed_dim)
        
        # 3. Concatenate into micro tier
        micro_tier = torch.cat([tab_proj, icd_micro_features], dim=-1) # (batch_size, 2 * embed_dim)
        
        # 4. Final output
        logits = self.head(micro_tier)
        return logits

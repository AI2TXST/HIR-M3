import unittest
import torch
from hier_icd_embedding import parse_icd_hierarchy, HierICDEmbedding, HIR_M3_ModelStub

class TestHierICDEmbedding(unittest.TestCase):

    def test_parse_icd_hierarchy(self):
        """Verify robust parsing of ICD-10 codes into (G0, G1, G2) levels."""
        # 1. Standard 3-char code
        g0, g1, g2 = parse_icd_hierarchy("I10")
        self.assertEqual((g0, g1, g2), ("I", "I1", "I10"))

        # 2. Code with decimal
        g0, g1, g2 = parse_icd_hierarchy("E11.9")
        self.assertEqual((g0, g1, g2), ("E", "E1", "E11"))

        # 3. Another code with decimal
        g0, g1, g2 = parse_icd_hierarchy("J44.9")
        self.assertEqual((g0, g1, g2), ("J", "J4", "J44"))

        # 4. Long code with decimals
        g0, g1, g2 = parse_icd_hierarchy("C50.911")
        self.assertEqual((g0, g1, g2), ("C", "C5", "C50"))

        # 5. Padding / empty handling
        self.assertEqual(parse_icd_hierarchy("<PAD>"), ("<PAD>", "<PAD>", "<PAD>"))
        self.assertEqual(parse_icd_hierarchy(""), ("<PAD>", "<PAD>", "<PAD>"))

    def test_prefix_sharing_g0_g1(self):
        """Verify that codes sharing prefixes (e.g. 'I10' and 'I11') map to the exact same G0 and G1 IDs."""
        toy_codes = ["I10", "I11", "E11.9", "J44.9"]
        emb_layer = HierICDEmbedding(toy_codes, embed_dim=32)

        id_i10 = emb_layer.code_to_id["I10"]
        id_i11 = emb_layer.code_to_id["I11"]

        # Check G0 mapping (both share 'I')
        g0_i10 = emb_layer.code_to_g0[id_i10].item()
        g0_i11 = emb_layer.code_to_g0[id_i11].item()
        self.assertEqual(g0_i10, g0_i11, "I10 and I11 must share the exact same G0 ID ('I')!")

        # Check G1 mapping (both share 'I1')
        g1_i10 = emb_layer.code_to_g1[id_i10].item()
        g1_i11 = emb_layer.code_to_g1[id_i11].item()
        self.assertEqual(g1_i10, g1_i11, "I10 and I11 must share the exact same G1 ID ('I1')!")

        # Check G2 mapping (they should differ: 'I10' vs 'I11')
        g2_i10 = emb_layer.code_to_g2[id_i10].item()
        g2_i11 = emb_layer.code_to_g2[id_i11].item()
        self.assertNotEqual(g2_i10, g2_i11, "I10 and I11 must have different G2 IDs ('I10' vs 'I11')!")

    def test_shapes_and_forward(self):
        """Verify output tensor shapes for embeddings and pooled episode vectors."""
        toy_codes = ["I10", "I11", "E11.9", "J44.9"]
        emb_layer = HierICDEmbedding(toy_codes, embed_dim=32)

        # Batch of 2 patient episodes:
        # Episode 1: ["I10", "E11.9"]
        # Episode 2: ["J44.9", "<PAD>"]
        batch_strings = [["I10", "E11.9"], ["J44.9"]]
        code_tensor = emb_layer.encode_code_list(batch_strings)
        
        self.assertEqual(code_tensor.shape, (2, 2))

        # Forward pass: (Batch=2, NumCodes=2, EmbedDim=32)
        code_embs = emb_layer(code_tensor)
        self.assertEqual(code_embs.shape, (2, 2, 32))

        # Episode-level mean pooling: (Batch=2, EmbedDim=32)
        pooled = emb_layer.pool_episode_embeddings(code_embs, code_tensor, method="mean")
        self.assertEqual(pooled.shape, (2, 32))
        
        # Verify padding token (index 0) was masked out in episode 2
        self.assertFalse(torch.isnan(pooled).any())

    def test_hir_m3_integration_stub(self):
        """Verify integration stub with HIR-M3 model forward pass."""
        toy_codes = ["I10", "I11", "E11.9", "J44.9"]
        model = HIR_M3_ModelStub(num_tabular_features=10, code_list=toy_codes, embed_dim=32, hidden_dim=64)
        
        x_tab = torch.randn(4, 10)  # Batch of 4, 10 tabular features
        x_icd = torch.randint(0, len(toy_codes), (4, 3))  # Batch of 4, 3 ICD codes per episode
        
        logits = model(x_tab, x_icd)
        self.assertEqual(logits.shape, (4, 1))

if __name__ == "__main__":
    unittest.main()

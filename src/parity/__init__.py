"""
ACT-Parity v2: Adaptive Cross-Tier Parity Framework for Clinical Readmission Equity.
Provides tokenized multi-tier cross-attention, augmented Lagrangian D-GAP loss, 
harm-weighted equity metrics, and rigorous factorial ablation evaluation.
"""

from .models import ACTParityV2, get_m3_feature_groups_tokenized
from .loss import AugmentedLagrangianDGAPLoss, DualMultiplierManager
from .metrics import (
    calculate_stabilized_ctdi,
    calculate_harm_weighted_efnhi,
    calculate_platt_calibration_slope,
    evaluate_pareto_frontier,
    calculate_bootstrap_confidence_intervals,
    run_scsa_sensitivity_analysis
)
from .equity_metrics import (
    calculate_subgroup_metrics,
    calculate_equity_metrics,
    audit_dataset_equity,
    calculate_generalized_entropy_index,
    calculate_hirm3_attention_metrics,
    calculate_subgroup_attention_divergence
)

__version__ = "2.0.0"

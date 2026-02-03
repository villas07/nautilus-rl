"""
ML Institutional - Técnicas de nivel hedge fund/prop trading.

Basado en Marcos López de Prado - Advances in Financial Machine Learning.

Componentes:
- triple_barrier.py: Triple Barrier Labeling Method (R-005) [DONE]
- purged_kfold.py: Purged K-Fold Cross-Validation (R-006) [DONE]
- sample_weights.py: Sample weighting por uniqueness (R-007) [DONE]
- microstructure_features.py: Features de microestructura (R-008) [DONE]
- entropy_features.py: Features de entropía (R-008) [DONE]
- regime_detector.py: Regime Detection System (R-018) [DONE]
- agent_selector.py: Regime-Aware Agent Selector (R-018) [DONE]
- meta_labeling.py: Meta-Labeling for Signal Filtering (R-009) [DONE]
- kelly_criterion.py: Kelly Criterion Bet Sizing (R-010) [DONE]

Reference: governance/evaluations/EVAL-001_institutional_ml.md, EVAL-003
"""

from .triple_barrier import TripleBarrierLabeling, TripleBarrierConfig
from .purged_kfold import PurgedKFold, cv_score
from .sample_weights import get_sample_weights, get_average_uniqueness
from .microstructure_features import (
    MicrostructureFeatures,
    MicrostructureConfig,
    get_microstructure_features,
)
from .entropy_features import (
    EntropyFeatures,
    EntropyConfig,
    get_entropy_features,
    get_multiscale_entropy_features,
)
from .regime_detector import (
    RegimeDetector,
    RegimeDetectorConfig,
    RegimeState,
    MarketRegime,
    detect_regime,
    get_regime_features,
)
from .agent_selector import (
    AgentSelector,
    AgentSelectorConfig,
    AgentConfig,
    EnsemblePrediction,
)
from .meta_labeling import (
    MetaLabeler,
    MetaLabelConfig,
    MetaLabel,
    MetaLabelingResult,
    MetaLabelingPipeline,
    PrimaryModelInterface,
    SignalType,
    create_meta_labels_from_triple_barrier,
)
from .kelly_criterion import (
    KellyCriterion,
    KellyConfig,
    KellyResult,
    KellyMethod,
    calculate_kelly_size,
    optimal_f_from_trades,
)

__all__ = [
    # Triple Barrier (R-005)
    "TripleBarrierLabeling",
    "TripleBarrierConfig",
    # Purged K-Fold (R-006)
    "PurgedKFold",
    "cv_score",
    # Sample Weights (R-007)
    "get_sample_weights",
    "get_average_uniqueness",
    # Microstructure Features (R-008)
    "MicrostructureFeatures",
    "MicrostructureConfig",
    "get_microstructure_features",
    # Entropy Features (R-008)
    "EntropyFeatures",
    "EntropyConfig",
    "get_entropy_features",
    "get_multiscale_entropy_features",
    # Regime Detection (R-018)
    "RegimeDetector",
    "RegimeDetectorConfig",
    "RegimeState",
    "MarketRegime",
    "detect_regime",
    "get_regime_features",
    # Agent Selector (R-018)
    "AgentSelector",
    "AgentSelectorConfig",
    "AgentConfig",
    "EnsemblePrediction",
    # Meta-Labeling (R-009)
    "MetaLabeler",
    "MetaLabelConfig",
    "MetaLabel",
    "MetaLabelingResult",
    "MetaLabelingPipeline",
    "PrimaryModelInterface",
    "SignalType",
    "create_meta_labels_from_triple_barrier",
    # Kelly Criterion (R-010)
    "KellyCriterion",
    "KellyConfig",
    "KellyResult",
    "KellyMethod",
    "calculate_kelly_size",
    "optimal_f_from_trades",
]

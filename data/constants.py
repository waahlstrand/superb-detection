from dataclasses import dataclass

THORACIC = [
    "T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4"
]

LUMBAR = [
    "L4", "L3", "L2", "L1"
]

VERTEBRA_NAMES = [
    *LUMBAR,
    *THORACIC
]

@dataclass
class SuperbStatistics:
    MEAN: float = 2048.724442444413
    STD: float = 375.3545219751161
    MAX: float = 4000.0
    MAX_HEIGHT: int = 1656
    MAX_WIDTH: int = 603
    MIN_HEIGHT: int = 1248
    MIN_WIDTH: int = 567

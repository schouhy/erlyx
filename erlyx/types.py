from typing import NamedTuple, List, Any, Optional, Dict
import numpy as np

DiscreteDistributionType = np.ndarray
ActionType = Any
ObservationType = Any
RewardType = float


class EpisodeStatus(NamedTuple):
    observation: ObservationType
    reward: RewardType
    done: bool

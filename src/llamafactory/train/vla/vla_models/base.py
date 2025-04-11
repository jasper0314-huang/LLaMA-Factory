from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch


@dataclass
class Base_VLAOutput(ModelOutput):
    cognition_features: torch.FloatTensor = None


class Base_VLAConfig(PretrainedConfig):
    def __init__(
        self,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        repeated_diffusion_steps: int = 4,
        **kwargs,
    ):
        self.action_dim = action_dim
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.norm_stats = norm_stats
        self.repeated_diffusion_steps = repeated_diffusion_steps
        super().__init__(**kwargs)

from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch


@dataclass
class Base_TraceOutput(ModelOutput):
    cognition_features: torch.FloatTensor = None


class Base_TraceConfig(PretrainedConfig):
    def __init__(
        self,
        num_trace_points: int = 16,
        **kwargs,
    ):
        self.num_trace_points = num_trace_points
        super().__init__(**kwargs)

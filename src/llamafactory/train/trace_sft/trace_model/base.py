from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch


@dataclass
class Base_TraceOutput(ModelOutput):
    traces: Optional[torch.Tensor] = None
    trace_loss: Optional[torch.Tensor] = None


class Base_TraceConfig(PretrainedConfig):
    def __init__(
        self,
        num_trace_points: int = 16,
        trace_loss_weight: float = 1.0,
        **kwargs,
    ):
        self.num_trace_points = num_trace_points
        self.trace_loss_weight = trace_loss_weight
        super().__init__(**kwargs)
"""Real-time Lane Detection for Autonomous Driving."""

from .model import create_lanenet, LaneNet
from .dataset import LaneDataset, create_dataloaders
from .inference import LaneInferenceEngine, VideoInferenceEngine
from .postprocess import LanePostProcessor, RealTimeLaneProcessor
from .metrics import LaneMetrics, PerformanceProfiler, BenchmarkComparison
from .train import Trainer

__version__ = '1.0.0'
__author__ = 'Lane Detection Team'

__all__ = [
    'create_lanenet',
    'LaneNet',
    'LaneDataset',
    'create_dataloaders',
    'LaneInferenceEngine',
    'VideoInferenceEngine',
    'LanePostProcessor',
    'RealTimeLaneProcessor',
    'LaneMetrics',
    'PerformanceProfiler',
    'BenchmarkComparison',
    'Trainer',
]

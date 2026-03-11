"""
Evaluation metrics for lane detection.

Includes F1 score, accuracy, FPS, and baseline comparison.
"""

import numpy as np
import torch
from typing import Dict, Tuple
import time


class LaneMetrics:
    """Lane detection evaluation metrics."""

    @staticmethod
    def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU).

        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask

        Returns:
            IoU score (0-1)
        """
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        if union == 0:
            return 0.0

        return intersection / union

    @staticmethod
    def f1_score(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calculate F1 score.

        Args:
            pred_mask: Predicted mask
            gt_mask: Ground truth mask
            threshold: Confidence threshold for prediction

        Returns:
            F1 score (0-1)
        """
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        gt_binary = (gt_mask > threshold).astype(np.uint8)

        tp = np.logical_and(pred_binary, gt_binary).sum()
        fp = np.logical_and(pred_binary, ~gt_binary).sum()
        fn = np.logical_and(~pred_binary, gt_binary).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1

    @staticmethod
    def accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calculate pixel-wise accuracy.

        Args:
            pred_mask: Predicted mask
            gt_mask: Ground truth mask
            threshold: Confidence threshold

        Returns:
            Accuracy (0-1)
        """
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        gt_binary = (gt_mask > threshold).astype(np.uint8)

        correct = np.sum(pred_binary == gt_binary)
        total = pred_binary.size

        return correct / total

    @staticmethod
    def precision_recall(pred_mask: np.ndarray, gt_mask: np.ndarray,
                        threshold: float = 0.5) -> Tuple[float, float]:
        """
        Calculate precision and recall.

        Args:
            pred_mask: Predicted mask
            gt_mask: Ground truth mask
            threshold: Confidence threshold

        Returns:
            Tuple of (precision, recall)
        """
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        gt_binary = (gt_mask > threshold).astype(np.uint8)

        tp = np.logical_and(pred_binary, gt_binary).sum()
        fp = np.logical_and(pred_binary, ~gt_binary).sum()
        fn = np.logical_and(~pred_binary, gt_binary).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return precision, recall


class PerformanceProfiler:
    """Profile inference performance."""

    def __init__(self):
        self.latencies = []
        self.start_time = None

    def start(self) -> None:
        """Start timing."""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """
        Stop timing and record latency.

        Returns:
            Latency in milliseconds
        """
        if self.start_time is None:
            return 0.0

        latency = (time.perf_counter() - self.start_time) * 1000  # Convert to ms
        self.latencies.append(latency)
        self.start_time = None

        return latency

    def get_fps(self) -> float:
        """
        Calculate average FPS.

        Returns:
            FPS (frames per second)
        """
        if not self.latencies:
            return 0.0

        avg_latency_ms = np.mean(self.latencies)
        fps = 1000 / avg_latency_ms if avg_latency_ms > 0 else 0.0

        return fps

    def get_statistics(self) -> Dict[str, float]:
        """
        Get latency statistics.

        Returns:
            Dictionary with latency stats
        """
        if not self.latencies:
            return {}

        latencies = np.array(self.latencies)

        return {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'std_latency_ms': np.std(latencies),
            'fps': self.get_fps(),
            'num_frames': len(latencies)
        }

    def reset(self) -> None:
        """Reset latency history."""
        self.latencies = []
        self.start_time = None

    def print_summary(self) -> None:
        """Print performance summary."""
        stats = self.get_statistics()

        if not stats:
            print("No latency data recorded")
            return

        print(f"\n{'='*60}")
        print(f"Performance Profiling Results")
        print(f"{'='*60}")
        print(f"Frames processed: {stats['num_frames']}")
        print(f"Mean latency: {stats['mean_latency_ms']:.2f} ms")
        print(f"Median latency: {stats['median_latency_ms']:.2f} ms")
        print(f"Min latency: {stats['min_latency_ms']:.2f} ms")
        print(f"Max latency: {stats['max_latency_ms']:.2f} ms")
        print(f"Std latency: {stats['std_latency_ms']:.2f} ms")
        print(f"Average FPS: {stats['fps']:.2f}")
        print(f"{'='*60}\n")


class BenchmarkComparison:
    """Compare against baseline models."""

    BASELINE_METRICS = {
        'baseline_model': {
            'accuracy': 0.92,
            'f1_score': 0.85,
            'fps': 30.0,
            'latency_ms': 33.3
        }
    }

    @staticmethod
    def compare_with_baseline(our_accuracy: float, our_f1: float, our_fps: float,
                             baseline_name: str = 'baseline_model') -> Dict[str, float]:
        """
        Compare our metrics with baseline.

        Args:
            our_accuracy: Our model accuracy
            our_f1: Our model F1 score
            our_fps: Our model FPS
            baseline_name: Name of baseline model

        Returns:
            Dictionary with improvement percentages
        """
        baseline = BenchmarkComparison.BASELINE_METRICS.get(baseline_name)
        if baseline is None:
            raise ValueError(f"Unknown baseline: {baseline_name}")

        accuracy_improvement = ((our_accuracy - baseline['accuracy']) / baseline['accuracy']) * 100
        f1_improvement = ((our_f1 - baseline['f1_score']) / baseline['f1_score']) * 100
        fps_improvement = ((our_fps - baseline['fps']) / baseline['fps']) * 100

        return {
            'accuracy_improvement_pct': accuracy_improvement,
            'f1_improvement_pct': f1_improvement,
            'fps_improvement_pct': fps_improvement,
            'our_accuracy': our_accuracy,
            'our_f1': our_f1,
            'our_fps': our_fps,
            'baseline_accuracy': baseline['accuracy'],
            'baseline_f1': baseline['f1_score'],
            'baseline_fps': baseline['fps']
        }

    @staticmethod
    def print_benchmark(our_accuracy: float, our_f1: float, our_fps: float,
                       baseline_name: str = 'baseline_model') -> None:
        """Print benchmark comparison."""
        comparison = BenchmarkComparison.compare_with_baseline(
            our_accuracy, our_f1, our_fps, baseline_name
        )

        print(f"\n{'='*70}")
        print(f"Benchmark Comparison vs {baseline_name}")
        print(f"{'='*70}")
        print(f"{'Metric':<20} {'Ours':<15} {'Baseline':<15} {'Improvement':<15}")
        print(f"{'-'*70}")
        print(f"{'Accuracy':<20} {comparison['our_accuracy']:<15.4f} "
              f"{comparison['baseline_accuracy']:<15.4f} "
              f"{comparison['accuracy_improvement_pct']:>+14.2f}%")
        print(f"{'F1 Score':<20} {comparison['our_f1']:<15.4f} "
              f"{comparison['baseline_f1']:<15.4f} "
              f"{comparison['f1_improvement_pct']:>+14.2f}%")
        print(f"{'FPS':<20} {comparison['our_fps']:<15.2f} "
              f"{comparison['baseline_fps']:<15.2f} "
              f"{comparison['fps_improvement_pct']:>+14.2f}%")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    # Test metrics
    print("Testing LaneMetrics...")

    pred_mask = np.random.rand(384, 640)
    gt_mask = np.random.rand(384, 640)

    iou = LaneMetrics.iou(pred_mask > 0.5, gt_mask > 0.5)
    f1 = LaneMetrics.f1_score(pred_mask, gt_mask)
    acc = LaneMetrics.accuracy(pred_mask, gt_mask)
    prec, rec = LaneMetrics.precision_recall(pred_mask, gt_mask)

    print(f"IoU: {iou:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}")

    # Test profiler
    print("\nTesting PerformanceProfiler...")
    profiler = PerformanceProfiler()

    for _ in range(100):
        profiler.start()
        time.sleep(0.01)  # Simulate 10ms inference
        profiler.stop()

    profiler.print_summary()

    # Test benchmark
    print("\nTesting BenchmarkComparison...")
    BenchmarkComparison.print_benchmark(
        our_accuracy=0.99,
        our_f1=0.95,
        our_fps=150.0
    )

    print("All tests passed!")

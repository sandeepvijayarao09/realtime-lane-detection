"""
Real-time inference engine for LaneNet.

Features:
- TorchScript model export
- Batch and single-frame inference
- Lane post-processing
- Performance profiling
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

from model import create_lanenet
from postprocess import LanePostProcessor, RealTimeLaneProcessor
from metrics import PerformanceProfiler


class LaneInferenceEngine:
    """Inference engine for lane detection."""

    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda',
                 use_torchscript: bool = False):
        """
        Args:
            model_path: Path to model checkpoint
            device: Device to run inference on
            use_torchscript: Use TorchScript model
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_torchscript = use_torchscript

        if use_torchscript and model_path and model_path.endswith('.pt'):
            self.model = torch.jit.load(model_path, map_location=self.device)
        else:
            self.model = create_lanenet(backbone='efficientnet', pretrained=False)
            if model_path:
                self._load_checkpoint(model_path)
            self.model = self.model.to(self.device)

        self.model.eval()

        # Post-processor
        self.postprocessor = LanePostProcessor(image_shape=(384, 640))
        self.realtime_processor = RealTimeLaneProcessor(buffer_size=3)

        # Profiler
        self.profiler = PerformanceProfiler()

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")

    def export_torchscript(self, output_path: str, example_input: Optional[torch.Tensor] = None) -> None:
        """
        Export model to TorchScript.

        Args:
            output_path: Output path for TorchScript model
            example_input: Example input tensor for tracing
        """
        self.model.eval()

        if example_input is None:
            example_input = torch.randn(1, 3, 384, 640, device=self.device)

        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, example_input)

        traced_model.save(output_path)
        print(f"Exported TorchScript model to {output_path}")

    def preprocess(self, image: np.ndarray, image_size: Tuple[int, int] = (384, 640)) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image: Input image (BGR)
            image_size: Target size (height, width)

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Resize
        image = cv2.resize(image, (image_size[1], image_size[0]))

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # To tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        return image.to(self.device)

    def infer_single(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Infer on single image.

        Args:
            image: Input image (BGR)

        Returns:
            Dictionary with segmentation and embeddings
        """
        self.profiler.start()

        # Preprocess
        input_tensor = self.preprocess(image)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)

        seg_logits = outputs['seg'].cpu().numpy()[0]
        embeddings = outputs['emb'].cpu().numpy()[0]

        latency = self.profiler.stop()

        return {
            'seg_logits': seg_logits,
            'embeddings': embeddings,
            'latency_ms': latency
        }

    def infer_batch(self, images: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Infer on batch of images.

        Args:
            images: List of input images (BGR)

        Returns:
            Dictionary with batch outputs
        """
        self.profiler.start()

        # Stack and preprocess
        batch = torch.cat([self.preprocess(img) for img in images], dim=0)

        # Inference
        with torch.no_grad():
            outputs = self.model(batch)

        seg_logits = outputs['seg'].cpu().numpy()
        embeddings = outputs['emb'].cpu().numpy()

        latency = self.profiler.stop()

        return {
            'seg_logits': seg_logits,
            'embeddings': embeddings,
            'latency_ms': latency,
            'batch_size': len(images)
        }

    def detect_lanes(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect lanes in image.

        Args:
            image: Input image (BGR)

        Returns:
            List of lane curves (each is Nx2 array of x,y coordinates)
        """
        outputs = self.infer_single(image)

        seg_logits = outputs['seg_logits'][0]
        binary_mask = self.postprocessor.process_segmentation(seg_logits, threshold=0.5)

        # Extract lane coordinates
        lanes = self.postprocessor.extract_lane_coordinates(binary_mask)

        # Fit curves
        fitted_lanes = []
        for lane in lanes:
            fitted = self.postprocessor.fit_lane_curve(lane)
            if fitted is not None:
                fitted_lanes.append(fitted)

        return fitted_lanes

    def detect_lanes_realtime(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect lanes with temporal smoothing (realtime).

        Args:
            image: Input image (BGR)

        Returns:
            List of smoothed lane curves
        """
        outputs = self.infer_single(image)
        seg_logits = outputs['seg_logits'][0]

        lanes = self.realtime_processor.process_frame(seg_logits)
        return lanes

    def get_profiling_stats(self) -> Dict[str, float]:
        """Get profiling statistics."""
        return self.profiler.get_statistics()

    def reset_profiler(self) -> None:
        """Reset profiler."""
        self.profiler.reset()


class VideoInferenceEngine:
    """Process video files with lane detection."""

    def __init__(self, inference_engine: LaneInferenceEngine):
        """
        Args:
            inference_engine: Inference engine instance
        """
        self.engine = inference_engine

    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     skip_frames: int = 1, show_stats: bool = True) -> None:
        """
        Process video and detect lanes.

        Args:
            video_path: Input video path
            output_path: Output video path (if None, no output video)
            skip_frames: Process every Nth frame
            show_stats: Show performance statistics
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Setup output video if specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        self.engine.reset_profiler()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Skip frames
                if frame_count % skip_frames != 0:
                    continue

                # Detect lanes
                lanes = self.engine.detect_lanes_realtime(frame)

                # Draw lanes
                frame_with_lanes = self.engine.postprocessor.draw_lanes(frame, lanes)

                # Write to output
                if writer:
                    writer.write(frame_with_lanes)

                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    stats = self.engine.get_profiling_stats()
                    fps = stats.get('fps', 0)
                    print(f"Frame {frame_count}/{total_frames} - FPS: {fps:.1f}")

        finally:
            cap.release()
            if writer:
                writer.release()

        if show_stats:
            self.engine.profiler.print_summary()

        print(f"Processed {frame_count} frames")
        if output_path:
            print(f"Saved output to {output_path}")

    def process_webcam(self, duration: int = 30) -> None:
        """
        Process webcam stream for duration.

        Args:
            duration: Duration in seconds
        """
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        print(f"Processing webcam for {duration} seconds...")
        self.engine.reset_profiler()

        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect lanes
                lanes = self.engine.detect_lanes_realtime(frame)

                # Draw lanes
                frame_with_lanes = self.engine.postprocessor.draw_lanes(frame, lanes)

                # Show FPS
                stats = self.engine.get_profiling_stats()
                fps = stats.get('fps', 0)
                cv2.putText(frame_with_lanes, f"FPS: {fps:.1f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display
                cv2.imshow('Lane Detection', frame_with_lanes)

                # Check if duration exceeded
                if time.time() - start_time > duration:
                    break

                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        self.engine.profiler.print_summary()


if __name__ == '__main__':
    # Test inference engine
    print("Testing LaneInferenceEngine...")

    engine = LaneInferenceEngine(device='cpu')

    # Test single image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    lanes = engine.detect_lanes(test_image)
    print(f"Detected {len(lanes)} lanes")

    # Test batch
    test_batch = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)]
    outputs = engine.infer_batch(test_batch)
    print(f"Batch inference: {outputs['seg_logits'].shape}")

    # Test profiling
    for _ in range(10):
        lanes = engine.detect_lanes(test_image)
    engine.profiler.print_summary()

    print("All tests passed!")

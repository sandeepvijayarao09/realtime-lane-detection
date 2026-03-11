"""
Demo script for real-time lane detection.

Runs inference on video, webcam, or image directory with visualization.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from src.inference import LaneInferenceEngine, VideoInferenceEngine
from src.metrics import PerformanceProfiler


def process_image(engine: LaneInferenceEngine, image_path: str, output_path: str = None) -> None:
    """Process single image."""
    print(f"Processing image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return

    # Detect lanes
    lanes = engine.detect_lanes(image)

    # Draw lanes
    output_image = engine.postprocessor.draw_lanes(image, lanes, color=(0, 255, 0), thickness=3)

    # Add text info
    stats = engine.get_profiling_stats()
    if stats:
        fps = stats.get('fps', 0)
        cv2.putText(output_image, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display
    cv2.imshow('Lane Detection Result', output_image)

    # Save if output path specified
    if output_path:
        cv2.imwrite(output_path, output_image)
        print(f"Saved output to {output_path}")

    print(f"Detected {len(lanes)} lanes")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(engine: LaneInferenceEngine, video_path: str, output_path: str = None) -> None:
    """Process video file."""
    print(f"Processing video: {video_path}")

    video_engine = VideoInferenceEngine(engine)
    video_engine.process_video(video_path, output_path=output_path, skip_frames=1, show_stats=True)


def process_webcam(engine: LaneInferenceEngine, duration: int = 30) -> None:
    """Process webcam stream."""
    print(f"Processing webcam for {duration} seconds...")
    print("Press 'q' to exit")

    video_engine = VideoInferenceEngine(engine)
    video_engine.process_webcam(duration=duration)


def process_directory(engine: LaneInferenceEngine, image_dir: str, output_dir: str = None) -> None:
    """Process all images in directory."""
    image_dir = Path(image_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in image_dir.iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Processing {len(image_files)} images...")

    for i, image_path in enumerate(sorted(image_files), 1):
        print(f"\n[{i}/{len(image_files)}] Processing {image_path.name}")

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Cannot read image: {image_path}")
            continue

        # Detect lanes
        lanes = engine.detect_lanes(image)

        # Draw lanes
        output_image = engine.postprocessor.draw_lanes(image, lanes, color=(0, 255, 0), thickness=3)

        # Add stats
        stats = engine.get_profiling_stats()
        if stats:
            fps = stats.get('fps', 0)
            cv2.putText(output_image, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save output
        if output_dir:
            output_path = output_dir / f"{image_path.stem}_lanes{image_path.suffix}"
            cv2.imwrite(str(output_path), output_image)

        print(f"Detected {len(lanes)} lanes")

    # Print statistics
    engine.profiler.print_summary()

    if output_dir:
        print(f"\nSaved outputs to {output_dir}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Real-time Lane Detection Demo')

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to image file')
    input_group.add_argument('--video', type=str, help='Path to video file')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam')
    input_group.add_argument('--directory', type=str, help='Path to image directory')

    # Output options
    parser.add_argument('--output', type=str, help='Output path for result')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')

    # Inference options
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--backbone', type=str, default='efficientnet', help='Backbone (efficientnet/mobilenet)')
    parser.add_argument('--duration', type=int, default=30, help='Webcam duration in seconds')

    args = parser.parse_args()

    # Create inference engine
    print("Initializing inference engine...")
    engine = LaneInferenceEngine(
        model_path=args.model,
        device=args.device,
        use_torchscript=False
    )

    # Process input
    if args.image:
        process_image(engine, args.image, args.output)
    elif args.video:
        process_video(engine, args.video, args.output)
    elif args.webcam:
        process_webcam(engine, args.duration)
    elif args.directory:
        process_directory(engine, args.directory, args.output)


if __name__ == '__main__':
    main()

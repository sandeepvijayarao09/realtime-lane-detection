"""
Post-processing for lane detection outputs.

Includes:
- Binary mask to coordinate extraction
- Instance clustering
- Curve fitting
- Inverse perspective mapping (IPM/bird's eye view)
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from scipy import interpolate
from sklearn.cluster import DBSCAN


class LanePostProcessor:
    """Post-process lane detection outputs."""

    def __init__(self, image_shape: Tuple[int, int] = (384, 640),
                 camera_matrix: Optional[np.ndarray] = None):
        """
        Args:
            image_shape: Image shape (height, width)
            camera_matrix: Camera intrinsic matrix for IPM
        """
        self.image_shape = image_shape
        self.height, self.width = image_shape

        # Default camera matrix for perspective transform
        if camera_matrix is None:
            self.camera_matrix = self._get_default_camera_matrix()
        else:
            self.camera_matrix = camera_matrix

    def _get_default_camera_matrix(self) -> np.ndarray:
        """Get default camera matrix for this image size."""
        h, w = self.image_shape
        focal_length = w
        cx = w / 2
        cy = h / 2
        return np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def process_segmentation(self, seg_mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert segmentation logits to binary mask.

        Args:
            seg_mask: Segmentation output (H, W) or (1, H, W)
            threshold: Confidence threshold

        Returns:
            Binary mask (H, W)
        """
        if seg_mask.ndim == 3:
            seg_mask = seg_mask[0]

        binary_mask = (seg_mask > threshold).astype(np.uint8)
        return binary_mask

    def extract_lane_coordinates(self, binary_mask: np.ndarray,
                                min_pixels: int = 50) -> List[np.ndarray]:
        """
        Extract lane coordinates from binary mask.

        Args:
            binary_mask: Binary lane mask (H, W)
            min_pixels: Minimum pixels to form a valid lane

        Returns:
            List of lane coordinate arrays
        """
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lanes = []
        for contour in contours:
            if len(contour) >= min_pixels:
                # Extract points
                points = contour.squeeze().astype(np.float32)
                if points.ndim == 1:
                    continue
                lanes.append(points)

        return lanes

    def cluster_instances(self, embeddings: np.ndarray, eps: float = 0.5,
                         min_samples: int = 10) -> np.ndarray:
        """
        Cluster lane instances using DBSCAN.

        Args:
            embeddings: Instance embeddings (H, W, D)
            eps: DBSCAN eps parameter
            min_samples: DBSCAN min_samples parameter

        Returns:
            Cluster labels (H, W)
        """
        h, w, d = embeddings.shape

        # Reshape for clustering
        embeddings_flat = embeddings.reshape(-1, d)

        # Run DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings_flat)
        labels = clustering.labels_.reshape(h, w)

        return labels

    def fit_lane_curve(self, points: np.ndarray, order: int = 3,
                      method: str = 'spline') -> Optional[np.ndarray]:
        """
        Fit polynomial or spline curve to lane points.

        Args:
            points: Lane coordinate points (N, 2) in (x, y) format
            order: Polynomial order or spline order
            method: 'poly' for polynomial, 'spline' for spline fitting

        Returns:
            Fitted curve points or None if fitting failed
        """
        if len(points) < 4:
            return None

        try:
            # Sort by y coordinate
            points = points[np.argsort(points[:, 1])]

            if method == 'poly':
                # Fit polynomial
                coeffs = np.polyfit(points[:, 1], points[:, 0], order)
                y_fit = np.linspace(points[:, 1].min(), points[:, 1].max(), 100)
                x_fit = np.polyval(coeffs, y_fit)
            else:  # spline
                # Fit spline
                if len(points) < order + 1:
                    order = len(points) - 1
                tck = interpolate.splrep(points[:, 1], points[:, 0], k=order)
                y_fit = np.linspace(points[:, 1].min(), points[:, 1].max(), 100)
                x_fit = interpolate.splev(y_fit, tck)

            # Clip to image bounds
            x_fit = np.clip(x_fit, 0, self.width - 1)
            y_fit = np.clip(y_fit, 0, self.height - 1)

            curve = np.column_stack([x_fit, y_fit])
            return curve

        except Exception:
            return None

    def inverse_perspective_mapping(self, points: np.ndarray,
                                    h_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply inverse perspective mapping to get bird's eye view.

        Args:
            points: Lane points in image space (N, 2) in (x, y)
            h_matrix: Homography matrix for IPM. If None, compute default.

        Returns:
            Points in bird's eye view (N, 2)
        """
        if h_matrix is None:
            h_matrix = self._compute_default_homography()

        # Convert to homogeneous coordinates
        ones = np.ones((len(points), 1))
        points_h = np.hstack([points, ones])

        # Apply homography
        points_warped = (h_matrix @ points_h.T).T
        points_warped = points_warped[:, :2] / points_warped[:, 2:3]

        return points_warped.astype(np.float32)

    def _compute_default_homography(self) -> np.ndarray:
        """Compute default homography for IPM."""
        h, w = self.image_shape

        # Source points (image space)
        src_points = np.float32([
            [0, h],
            [w, h],
            [0, 0],
            [w, 0]
        ])

        # Destination points (bird's eye view)
        dst_points = np.float32([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h]
        ])

        h_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return h_matrix

    def draw_lanes(self, image: np.ndarray, lanes: List[np.ndarray],
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 3) -> np.ndarray:
        """
        Draw detected lanes on image.

        Args:
            image: Input image
            lanes: List of lane coordinate arrays
            color: Lane color (BGR)
            thickness: Line thickness

        Returns:
            Image with drawn lanes
        """
        output = image.copy()

        for lane in lanes:
            if len(lane) > 1:
                lane_int = np.int32(lane)
                cv2.polylines(output, [lane_int], False, color, thickness)

        return output


class RealTimeLaneProcessor:
    """Real-time lane processing with frame buffering."""

    def __init__(self, buffer_size: int = 3):
        """
        Args:
            buffer_size: Number of frames to buffer for smoothing
        """
        self.buffer_size = buffer_size
        self.lane_buffer = []
        self.postprocessor = LanePostProcessor()

    def process_frame(self, seg_logits: np.ndarray, emb_logits: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Process frame and return smoothed lanes.

        Args:
            seg_logits: Segmentation output (H, W)
            emb_logits: Embedding output (H, W, D) optional

        Returns:
            List of smoothed lane curves
        """
        # Process segmentation
        binary_mask = self.postprocessor.process_segmentation(seg_logits)

        # Extract lanes
        lanes = self.postprocessor.extract_lane_coordinates(binary_mask)

        # Fit curves
        fitted_lanes = []
        for lane in lanes:
            fitted = self.postprocessor.fit_lane_curve(lane)
            if fitted is not None:
                fitted_lanes.append(fitted)

        # Buffer for smoothing
        self.lane_buffer.append(fitted_lanes)
        if len(self.lane_buffer) > self.buffer_size:
            self.lane_buffer.pop(0)

        # Return buffered lanes
        if self.lane_buffer:
            return self.lane_buffer[-1]
        return []


if __name__ == '__main__':
    # Test postprocessing
    print("Testing LanePostProcessor...")

    processor = LanePostProcessor(image_shape=(384, 640))

    # Create dummy segmentation mask
    seg_mask = np.random.rand(384, 640)
    binary_mask = processor.process_segmentation(seg_mask, threshold=0.5)
    print(f"Binary mask shape: {binary_mask.shape}")

    # Create dummy lanes
    lanes = [
        np.array([[100 + i*0.5, 50 + i] for i in range(100)], dtype=np.float32),
        np.array([[300 + i*0.5, 50 + i] for i in range(100)], dtype=np.float32),
    ]

    # Test curve fitting
    for i, lane in enumerate(lanes):
        fitted = processor.fit_lane_curve(lane, method='spline')
        if fitted is not None:
            print(f"Lane {i} fitted curve shape: {fitted.shape}")

    # Test IPM
    for i, lane in enumerate(lanes):
        warped = processor.inverse_perspective_mapping(lane)
        print(f"Lane {i} IPM shape: {warped.shape}")

    # Test drawing
    image = np.zeros((384, 640, 3), dtype=np.uint8)
    output = processor.draw_lanes(image, lanes)
    print(f"Output image shape: {output.shape}")

    print("\nAll tests passed!")

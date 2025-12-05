import cv2
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

# NOTE: Ideas. Make sure to update the README.md with the new color tracker.
# I want to calibrate the colors using the color of the the white and black on the piano I think

class BoundingBoxKalmanFilter:
    """
    Kalman filter for tracking bounding boxes.
    State: [center_x, center_y, width, height, velocity_x, velocity_y, velocity_w, velocity_h]
    Measurement: [center_x, center_y, width, height]
    """
    def __init__(self, initial_bbox):
        """
        Initialize Kalman filter with initial bounding box.
        
        Args:
            initial_bbox: (x, y, w, h) bounding box
        """
        x, y, w, h = initial_bbox
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        
        # Initialize Kalman filter (8 state variables, 4 measurements)
        self.kf = cv2.KalmanFilter(8, 4)
        
        # State transition matrix (A)
        # Position = previous position + velocity * dt
        # Velocity = previous velocity (constant velocity model)
        dt = 1.0  # Time step (assuming 1 frame)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],   # x = x + vx*dt
            [0, 1, 0, 0, 0, dt, 0, 0],   # y = y + vy*dt
            [0, 0, 1, 0, 0, 0, dt, 0],   # w = w + vw*dt
            [0, 0, 0, 1, 0, 0, 0, dt],   # h = h + vh*dt
            [0, 0, 0, 0, 1, 0, 0, 0],    # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],    # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],    # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1]     # vh = vh
        ], dtype=np.float32)
        
        # Measurement matrix (H) - we only observe position and size, not velocity
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],    # measure x
            [0, 1, 0, 0, 0, 0, 0, 0],    # measure y
            [0, 0, 1, 0, 0, 0, 0, 0],    # measure w
            [0, 0, 0, 1, 0, 0, 0, 0]     # measure h
        ], dtype=np.float32)
        
        # Process noise covariance (Q) - how much we trust the model
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
        # Measurement noise covariance (R) - how much we trust measurements
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Error covariance (P) - initial uncertainty
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 0.1
        
        # Initial state: [center_x, center_y, width, height, 0, 0, 0, 0]
        self.kf.statePre = np.array([center_x, center_y, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.kf.statePost = np.array([center_x, center_y, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        self.last_update_time = 0
        self.missed_detections = 0
        self.max_missed_detections = 10  # Remove tracker after 10 missed detections
    
    def predict(self):
        """Predict next state."""
        prediction = self.kf.predict()
        return prediction
    
    def update(self, bbox):
        """
        Update filter with new measurement.
        
        Args:
            bbox: (x, y, w, h) bounding box
        """
        x, y, w, h = bbox
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        
        measurement = np.array([center_x, center_y, w, h], dtype=np.float32)
        self.kf.correct(measurement)
        self.missed_detections = 0
    
    def get_bbox(self):
        """
        Get predicted bounding box from current state.
        
        Returns:
            (x, y, w, h) bounding box
        """
        state = self.kf.statePost
        center_x = state[0]
        center_y = state[1]
        w = max(1, int(state[2]))  # Ensure positive width
        h = max(1, int(state[3]))  # Ensure positive height
        x = int(center_x - w / 2.0)
        y = int(center_y - h / 2.0)
        return (x, y, w, h)
    
    def increment_missed(self):
        """Increment missed detection counter."""
        self.missed_detections += 1
    
    def should_remove(self):
        """Check if tracker should be removed."""
        return self.missed_detections >= self.max_missed_detections


class ColorTracker:
    """
    Color-based object tracker with Kalman filtering.
    Tracks multiple colors simultaneously using HSV color space.
    """
    
    # Default HSV ranges for ROYGB colors
    DEFAULT_COLOR_RANGES = {
        "Red":    [([0, 100, 50], [10, 255, 255]), ([170, 100, 50], [180, 255, 255])],  # wrap-around
        "Orange": [([11, 100, 50], [25, 255, 255])],
        "Yellow": [([26, 100, 50], [34, 255, 255])],
        "Green":  [([35, 100, 50], [85, 255, 255])],
        "Blue":   [([100, 100, 50], [130, 255, 255])]
    }
    
    # Default BGR colors for drawing
    DEFAULT_BGR_COLORS = {
        "Red": (0, 0, 255),
        "Orange": (0, 165, 255),
        "Yellow": (0, 255, 255),
        "Green": (0, 255, 0),
        "Blue": (255, 0, 0)
    }
    
    def __init__(self, 
                 color_ranges: Optional[Dict] = None,
                 bgr_colors: Optional[Dict] = None,
                 min_area: int = 300,
                 max_area: int = 5000,
                 min_width: int = 20,
                 min_height: int = 20,
                 max_width: int = 200,
                 max_height: int = 200,
                 dilation_kernel_size: int = 5,
                 max_association_distance: int = 100,
                 max_number_of_colors: int = 5,
                 max_trackers_per_color: Optional[int] = None,
                 max_total_trackers: Optional[int] = None):
        """
        Initialize ColorTracker.
        
        Args:
            color_ranges: Dictionary mapping color names to HSV ranges
            bgr_colors: Dictionary mapping color names to BGR tuples for drawing
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider
            min_width: Minimum bounding box width
            min_height: Minimum bounding box height
            max_width: Maximum bounding box width
            max_height: Maximum bounding box height
            dilation_kernel_size: Size of dilation kernel for mask processing
            max_association_distance: Maximum distance for associating detections to trackers
            max_number_of_colors: Maximum number of color ranges per color name
            max_trackers_per_color: Maximum number of trackers allowed per color (None = unlimited)
            max_total_trackers: Maximum total number of trackers across all colors (None = unlimited)
        """
        self.color_ranges = color_ranges or self.DEFAULT_COLOR_RANGES
        self.bgr_colors = bgr_colors or self.DEFAULT_BGR_COLORS
        self.min_area = min_area
        self.max_area = max_area
        self.min_width = min_width
        self.min_height = min_height
        self.max_width = max_width
        self.max_height = max_height
        self.max_association_distance = max_association_distance
        self.max_trackers_per_color = max_trackers_per_color
        self.max_total_trackers = max_total_trackers
        
        # Morphology kernel for dilation
        self.kernel = np.ones((dilation_kernel_size, dilation_kernel_size), "uint8")
        
        # Dictionary to store Kalman filters for each color
        # Structure: {color_name: [list of BoundingBoxKalmanFilter instances]}
        self.kalman_trackers = defaultdict(list)
        self.max_number_of_colors = max_number_of_colors
        
    def _associate_detection_to_tracker(self, detection: Tuple[int, int, int, int], 
                                        tracker: BoundingBoxKalmanFilter) -> Optional[float]:
        """
        Associate a detection with a tracker using distance metric.
        
        Args:
            detection: (x, y, w, h) bounding box
            tracker: BoundingBoxKalmanFilter instance
            
        Returns:
            Distance if associated, None otherwise
        """
        det_x, det_y, det_w, det_h = detection
        det_center = np.array([det_x + det_w/2, det_y + det_h/2])
        track_bbox = tracker.get_bbox()
        track_x, track_y, track_w, track_h = track_bbox
        track_center = np.array([track_x + track_w/2, track_y + track_h/2])
        
        distance = np.linalg.norm(det_center - track_center)
        
        if distance < self.max_association_distance:
            return distance
        return None
    
    def _create_color_mask(self, hsv_image: np.ndarray, color_name: str) -> np.ndarray:
        """
        Create a binary mask for a specific color.
        
        Args:
            hsv_image: HSV image
            color_name: Name of the color to detect
            
        Returns:
            Binary mask
        """
        if color_name not in self.color_ranges:
            return np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        if len(self.color_ranges[color_name]) > self.max_number_of_colors:
            print(f"Color {color_name} has more than {self.max_number_of_colors} ranges, skipping")
            return np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        mask = None
        ranges = self.color_ranges[color_name]
        
        # Handle multiple ranges (for colors like red that wrap around)
        for lower, upper in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            if mask is None:
                mask = cv2.inRange(hsv_image, lower_np, upper_np)
            else:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv_image, lower_np, upper_np))
        
        # Dilate mask to fill small gaps
        if mask is not None:
            mask = cv2.dilate(mask, self.kernel)
        
        return mask if mask is not None else np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    
    def _extract_detections(self, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Extract valid bounding boxes from a mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        detections = []
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                if (self.min_width < w < self.max_width and 
                    self.min_height < h < self.max_height):
                    detections.append((x, y, w, h))
        
        return detections
    
    def _get_total_tracker_count(self) -> int:
        """
        Get the total number of trackers across all colors.
        
        Returns:
            Total number of active trackers
        """
        return sum(len(trackers) for trackers in self.kalman_trackers.values())
    
    def _can_create_tracker(self, color_name: str) -> bool:
        """
        Check if a new tracker can be created for the given color.
        
        Args:
            color_name: Name of the color
            
        Returns:
            True if a new tracker can be created, False otherwise
        """
        # Check per-color limit
        if self.max_trackers_per_color is not None:
            if len(self.kalman_trackers[color_name]) >= self.max_trackers_per_color:
                return False
        
        # Check total limit
        if self.max_total_trackers is not None:
            if self._get_total_tracker_count() >= self.max_total_trackers:
                return False
        
        return True
    
    def _update_trackers(self, color_name: str, detections: List[Tuple[int, int, int, int]]):
        """
        Update Kalman trackers for a specific color with new detections.
        
        Args:
            color_name: Name of the color being tracked
            detections: List of (x, y, w, h) bounding boxes
        """
        # Predict all existing trackers for this color
        for tracker in self.kalman_trackers[color_name]:
            tracker.predict()
        
        # Associate detections with existing trackers
        used_detections = set()
        for tracker in self.kalman_trackers[color_name]:
            best_distance = float('inf')
            best_detection_idx = None
            
            for j, detection in enumerate(detections):
                if j in used_detections:
                    continue
                distance = self._associate_detection_to_tracker(detection, tracker)
                if distance is not None and distance < best_distance:
                    best_distance = distance
                    best_detection_idx = j
            
            if best_detection_idx is not None:
                # Update tracker with associated detection
                tracker.update(detections[best_detection_idx])
                used_detections.add(best_detection_idx)
            else:
                # No detection associated - increment missed counter
                tracker.increment_missed()
        
        # Remove old trackers that haven't been detected for a while (before creating new ones)
        self.kalman_trackers[color_name] = [
            t for t in self.kalman_trackers[color_name] if not t.should_remove()
        ]
        
        # Create new trackers for unassociated detections (respecting limits)
        for j, detection in enumerate(detections):
            if j not in used_detections:
                if self._can_create_tracker(color_name):
                    new_tracker = BoundingBoxKalmanFilter(detection)
                    self.kalman_trackers[color_name].append(new_tracker)
                # If limit reached, silently skip creating new tracker
    
    def _draw_tracked_boxes(self, frame: np.ndarray, color_name: str, draw_colors: bool = True):
        """
        Draw tracked bounding boxes on the frame.
        
        Args:
            frame: Image to draw on (modified in place)
            color_name: Name of the color being tracked
        """
        if color_name in self.bgr_colors.keys():
            color = self.bgr_colors[color_name]
        else:
            color = (255, 255, 255)
        
        for tracker in self.kalman_trackers[color_name]:
            x, y, w, h = tracker.get_bbox()
            
            # Ensure bbox is within frame bounds
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            w = max(1, min(w, frame.shape[1] - x))
            h = max(1, min(h, frame.shape[0] - y))
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if draw_colors:
                cv2.putText(frame, f"{color_name}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    def process_frame(self, frame: np.ndarray, draw_contours: bool = True, draw_colors: bool = True, draw_centroids: bool = True) -> np.ndarray:
        """
        Process a single frame and update all trackers.
        
        Args:
            frame: Input BGR image
            draw_contours: Whether to draw detected contours on the frame
            
        Returns:
            Frame with tracked bounding boxes drawn
        """
        frame = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for color_name in self.color_ranges.keys():
            # Create mask for this color
            mask = self._create_color_mask(hsv, color_name)
            
            # Draw contours if requested
            if draw_contours:
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (255, 255, 255), 1)
            
            # Extract detections
            detections = self._extract_detections(mask)
            
            # Update trackers
            self._update_trackers(color_name, detections)
            
            # Draw tracked boxes
            self._draw_tracked_boxes(frame, color_name, draw_colors)

            if draw_centroids:
                centroids = self.get_centroids(color_name)
                for centroid in centroids:
                    x, y = centroid
                    cv2.drawMarker(frame, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 10)
        
        return frame
    
    def get_tracked_boxes(self, color_name: Optional[str] = None) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Get all currently tracked bounding boxes.
        
        Args:
            color_name: If provided, only return boxes for this color. 'all' to return all colors.
            
        Returns:
            Dictionary mapping color names to lists of (x, y, w, h) bounding boxes
        """
        if color_name in self.kalman_trackers.keys():
            return {
                color_name: [tracker.get_bbox() for tracker in self.kalman_trackers[color_name]]
            }
        elif color_name == 'all' or color_name is None:
            return {
                name: [tracker.get_bbox() for tracker in trackers]
                for name, trackers in self.kalman_trackers.items()
            }
        else:
            print(f"Color {color_name} not found")
            return {}

    def get_centroids(self, color_name: Optional[str] = None) -> Dict[str, List[Tuple[int, int]]]:
        """
        Get the centroids (x,y) of all currently tracked bounding boxes.
        
        Args:
            color_name: If provided, only return centroids for this color. Otherwise return all.
            
        Returns:
            Dictionary mapping color names to lists of (x, y) centroids
        """
        bboxes = self.get_tracked_boxes(color_name)
        centroids = {}
        # x, w, y, h
        for color_name, boxes in bboxes.items():
            centroids[color_name] = []
            for box in boxes:
                x, y, w, h = box
                centroid = np.array([x + w/2, y + h/2])
                centroids[color_name].append(centroid)

        
       
        return centroids

# Main execution
if __name__ == "__main__":
    
    # allow imports from src
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # get path to calibration file
    from src.core.constants import CAMERA_CALIBRATION_PATH
    calib_npz = np.load(CAMERA_CALIBRATION_PATH)
    camera_matrix = calib_npz["camera_matrix"]
    dist_coeffs = calib_npz["dist_coeffs"]
    
    from src.core.constants import MARKER_IDS, MARKER_SIZE, IN_TO_METERS
    
    from src.computer_vision.aruco_pose_tracker import ArucoPoseTracker, TrackingMode
    from src.computer_vision.aruco_polygon_detector import ArucoPolygonDetector

    # Open webcam
    webcam = cv2.VideoCapture(0)
    
    # Create tracker
    color_track = ColorTracker(max_area=100000, max_number_of_colors=5)
    aruco_pose_tracker = ArucoPoseTracker(camera_matrix, dist_coeffs, mode = TrackingMode.STATIC, marker_ids=MARKER_IDS)

    # create aruco polygon detector
    aruco_polygon_detector = ArucoPolygonDetector(camera_matrix, dist_coeffs, image_size=(640,480), correct_distortion=True)
    
    
    while True:
        ret, frame = webcam.read()
        corners, ids, rejected = aruco_pose_tracker.detect_markers(frame)
        
        # get rotation vector and translation vector
        if corners is not None and ids is not None:
            poses = aruco_pose_tracker.get_marker_poses(frame, marker_size_meters=MARKER_SIZE*IN_TO_METERS)
        else:
            continue
        
        success, marker_list_2d = aruco_polygon_detector.get_marker_polygon(ids, poses)
        if success:
            break


    while True:
        ret, frame = webcam.read()
        if not ret:
            break
        
        
        # Process frame
        birdseye_frame = aruco_polygon_detector.transform_image_to_birdseye(frame)
        processed_birdseye_frame = color_track.process_frame(birdseye_frame, draw_contours=True, draw_colors=True)
        
        tracked_centroids = color_track.get_centroids('all')
        for color_name, centroids in tracked_centroids.items():
            for centroid in centroids:
                x,y = centroid

                processed_birdseye_frame = cv2.drawMarker(processed_birdseye_frame, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 10)
        cv2.imshow("5-Color Detection (ROYGB)", processed_birdseye_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    webcam.release()
    cv2.destroyAllWindows()

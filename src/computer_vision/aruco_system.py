import cv2 as cv
import numpy as np
import os
from typing import List, Tuple, Optional
import math
from enum import Enum

# Import the new modular classes
try:
    # Try relative imports first (when used as a package)
    from .aruco_marker_generator import ArucoMarkerGenerator
    from .aruco_pose_tracker import ArucoPoseTracker
    from .aruco_polygon_detector import ArucoPolygonDetector
except ImportError:
    # Fall back to absolute imports (when used as a script)
    from aruco_marker_generator import ArucoMarkerGenerator
    from aruco_pose_tracker import ArucoPoseTracker
    from aruco_polygon_detector import ArucoPolygonDetector

# Legacy compatibility
PaperSizes = {"LETTER": (8.5, 11.0)}


class ArucoMarkerSystem:
    """
    DEPRECATED AND LEGACY
    Comprehensive ArUco marker system that combines generation and pose tracking.
    
    This class maintains backward compatibility while using the new modular architecture.
    For new projects, consider using the specialized classes directly:
    - ArucoMarkerGenerator: For creating markers and sheets
    - ArucoPoseTracker: For live pose tracking
    - ArucoPolygonDetector: For polygon detection
    """
    
    def __init__(self, dictionary_type=cv.aruco.DICT_6X6_250, marker_ids: list = None):
        """
        Initialize the ArUco marker system.
        
        Args:
            dictionary_type: ArUco dictionary type (default: DICT_6X6_250)
            marker_ids: List of expected marker IDs (optional)
        """
        # Initialize the modular components
        self.generator = ArucoMarkerGenerator(dictionary_type)
        self.tracker = ArucoPoseTracker(dictionary_type, marker_ids)
        self.polygon_detector = ArucoPolygonDetector(self.tracker)
        
        # Legacy compatibility attributes
        self.dictionary = self.generator.dictionary
        self.detector_params = self.tracker.detector_params
        self.detector = self.tracker.detector
        self.marker_ids = marker_ids
        
        # Legacy pose filtering attributes (delegate to tracker)
        self.use_adaptive_filtering = self.tracker.use_adaptive_filtering
        self.enable_pose_filtering = self.tracker.enable_pose_filtering
        self.debug_filtering = self.tracker.debug_filtering
        self.enforce_z_axis_out = self.tracker.enforce_z_axis_out
        self.enable_moving_average = self.tracker.enable_moving_average
        self.filter_window_size = self.tracker.filter_window_size
        self.pose_history = self.tracker.pose_history
    
    # ========== DELEGATION METHODS FOR BACKWARD COMPATIBILITY ==========
    
    # Marker Generation Methods (delegate to ArucoMarkerGenerator)
    def generate_marker(self, marker_id: int, size: int = 200) -> np.ndarray:
        """Generate an ArUco marker. Delegates to ArucoMarkerGenerator."""
        return self.generator.generate_marker(marker_id, size)
    
    def save_marker(self, marker_id: int, filename: str, size: int = 200):
        """Save an ArUco marker to file. Delegates to ArucoMarkerGenerator."""
        return self.generator.save_marker(marker_id, filename, size)
    
    def create_marker_sheet(self, input_sheet: np.ndarray = None, 
                           paper_size_inches: Tuple[float, float] = (8.5, 11.0),  
                           dpi: int = 300, marker_size_inches: float = 1.0,   
                           marker_locations: List[Tuple[float, float]] = None,
                           marker_ids: List[int] = None, 
                           filename: str = "aruco_marker_sheet.png") -> str:
        """Create a marker sheet. Delegates to ArucoMarkerGenerator."""
        return self.generator.create_marker_sheet(
            input_sheet, paper_size_inches, dpi, marker_size_inches,
            marker_locations, marker_ids, filename
        )
    
    # Pose Tracking Methods (delegate to ArucoPoseTracker)
    def detect_markers(self, image: np.ndarray) -> Tuple[List, List, List]:
        """Detect ArUco markers. Delegates to ArucoPoseTracker."""
        return self.tracker.detect_markers(image)
    
    def draw_detected_markers(self, image: np.ndarray, corners: List, ids: List) -> np.ndarray:
        """Draw detected markers. Delegates to ArucoPoseTracker."""
        return self.tracker.draw_detected_markers(image, corners, ids)
    
    def get_marker_poses(self, image: np.ndarray, camera_matrix: np.ndarray, 
                        dist_coeffs: np.ndarray, marker_size_meters: float = 0.05, 
                        last_poses: Optional[list[dict] | dict[int, dict]] = None) -> list[dict]:
        """Get 3D poses of markers. Delegates to ArucoPoseTracker."""
        return self.tracker.get_marker_poses(image, camera_matrix, dist_coeffs, 
                                           marker_size_meters, last_poses)
    
    def estimate_pose_vecs(self, corners: List, ids: List, camera_matrix: np.ndarray, 
                          dist_coeffs: np.ndarray, marker_length: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate pose vectors. Delegates to ArucoPoseTracker."""
        return self.tracker.estimate_pose_vecs(corners, ids, camera_matrix, dist_coeffs, marker_length)
    
    def draw_axes(self, image: np.ndarray, camera_matrix: np.ndarray, 
                  dist_coeffs: np.ndarray, rvecs: np.ndarray, tvecs: np.ndarray, 
                  length: float = 0.1) -> np.ndarray:
        """Draw 3D axes. Delegates to ArucoPoseTracker."""
        return self.tracker.draw_axes(image, camera_matrix, dist_coeffs, rvecs, tvecs, length)
    
    def rotation_to_euler(self, rotation_matrix):
        """Convert rotation matrix to Euler angles. Delegates to ArucoPoseTracker."""
        return self.tracker.rotation_to_euler(rotation_matrix)
    
    def analyze_z_axis_orientation(self, pose: dict) -> dict:
        """Analyze Z-axis orientation. Delegates to ArucoPoseTracker."""
        return self.tracker.analyze_z_axis_orientation(pose)
    
    # Polygon Detection Methods (delegate to ArucoPolygonDetector)
    def get_marker_polygon(self, ordered_ids: List[int], image: np.ndarray, 
                          camera_matrix: np.ndarray, dist_coeffs: np.ndarray, 
                          marker_size_meters: float = 0.05) -> List:
        """Get marker polygon. Delegates to ArucoPolygonDetector."""
        return self.polygon_detector.get_marker_polygon(
            ordered_ids, image, camera_matrix, dist_coeffs, marker_size_meters
        )
    
    def configure_pose_filtering(self, enable_filtering: bool = True,
                               adaptive_thresholds: bool = True, 
                               debug_output: bool = False,
                               enforce_z_axis_out: bool = True,
                               enable_moving_average: bool = True,
                               filter_window_size: int = 5):
        """
        Configure pose filtering behavior. Delegates to ArucoPoseTracker.
        
        Args:
            enable_filtering: Enable/disable pose filtering entirely
            adaptive_thresholds: Use adaptive thresholds based on movement
            debug_output: Print debugging information
            enforce_z_axis_out: Force Z-axis to always point toward camera (negative Y component)
            enable_moving_average: Enable TRUE LTI moving average filtering
            filter_window_size: Number of samples for moving average (higher = smoother)
        """
        # Delegate to the tracker
        self.tracker.configure_pose_filtering(
            enable_filtering, adaptive_thresholds, debug_output, 
            enforce_z_axis_out, enable_moving_average, filter_window_size
        )
        
        # Update legacy attributes for backward compatibility
        self.enable_pose_filtering = self.tracker.enable_pose_filtering
        self.use_adaptive_filtering = self.tracker.use_adaptive_filtering
        self.debug_filtering = self.tracker.debug_filtering
        self.enforce_z_axis_out = self.tracker.enforce_z_axis_out
        self.enable_moving_average = self.tracker.enable_moving_average
        self.filter_window_size = self.tracker.filter_window_size
        
    def configure_pose_filtering(self, enable=True, window_size=5):
        """Configure pose filtering parameters."""
        self.tracker.configure_pose_filtering(
            enable_filtering=enable, 
            filter_window_size=window_size
        )
        # Update legacy attributes for backward compatibility
        self.enable_pose_filtering = self.tracker.enable_pose_filtering
        self.use_adaptive_filtering = self.tracker.use_adaptive_filtering
        self.debug_filtering = self.tracker.debug_filtering
        self.enforce_z_axis_out = self.tracker.enforce_z_axis_out
        self.enable_moving_average = self.tracker.enable_moving_average
        self.filter_window_size = self.tracker.filter_window_size
        
    def get_marker_polygon(self, frame, camera_matrix, dist_coeffs, target_ids=None):
        """Get polygon formed by ArUco markers."""
        return self.polygon_detector.get_marker_polygon(frame, camera_matrix, dist_coeffs, target_ids)
    
    # Delegation methods for marker generation functionality
    def generate_marker(self, marker_id: int, size: int = 200) -> np.ndarray:
        """Generate an ArUco marker."""
        return self.generator.generate_marker(marker_id, size)
    
    def save_marker(self, marker_id: int, filename: str, size: int = 200):
        """Generate and save an ArUco marker to file."""
        return self.generator.save_marker(marker_id, filename, size)
    
    def create_marker_sheet(self, input_sheet=None, paper_size_inches=(8.5, 11.0), 
                           dpi=300, marker_size_inches=1.0, marker_locations=None, 
                           marker_ids=None, filename="aruco_marker_sheet.png"):
        """Create a sheet with markers at specified locations."""
        return self.generator.create_marker_sheet(
            input_sheet, paper_size_inches, dpi, marker_size_inches,
            marker_locations, marker_ids, filename
        )
    
    # Delegation methods for pose tracking functionality  
    def detect_markers(self, image: np.ndarray):
        """Detect ArUco markers in an image."""
        return self.tracker.detect_markers(image)
    
    def draw_detected_markers(self, image: np.ndarray, corners, ids):
        """Draw detected markers on the image."""
        return self.tracker.draw_detected_markers(image, corners, ids)
    
    def estimate_pose_vecs(self, corners, ids, camera_matrix: np.ndarray, 
                          dist_coeffs: np.ndarray, marker_length: float):
        """Estimate pose of detected markers.""" 
        return self.tracker.estimate_pose_vecs(corners, ids, camera_matrix, dist_coeffs, marker_length)
    
    def draw_axes(self, image: np.ndarray, camera_matrix: np.ndarray, 
                  dist_coeffs: np.ndarray, rvecs: np.ndarray, tvecs: np.ndarray, 
                  length: float = 0.1):
        """Draw 3D axes on detected markers."""
        return self.tracker.draw_axes(image, camera_matrix, dist_coeffs, rvecs, tvecs, length)
    
    def get_marker_poses(self, image: np.ndarray, camera_matrix: np.ndarray, 
                        dist_coeffs: np.ndarray, marker_size_meters: float = 0.05, 
                        last_poses=None):
        """Get 3D pose of all detected markers."""
        return self.tracker.get_marker_poses(image, camera_matrix, dist_coeffs, marker_size_meters, last_poses)
    
    def calculate_pose_similarity(self, current_pose: dict, previous_pose: dict, 
                                 adaptive_thresholds: bool = True):
        """Calculate similarity metrics between poses."""
        return self.tracker.calculate_pose_similarity(current_pose, previous_pose, adaptive_thresholds)
    
    def rotation_to_euler(self, rotation_matrix):
        """Convert rotation matrix to Euler angles."""
        return self.tracker.rotation_to_euler(rotation_matrix)
    
    def apply_moving_average_filter(self, pose_info: dict):
        """Apply LTI moving average filter to pose."""
        return self.tracker.apply_moving_average_filter(pose_info)
    
    def verify_lti_properties(self, marker_id: int):
        """Verify LTI properties of the filter."""
        return self.tracker.verify_lti_properties(marker_id)
    
    def correct_z_axis_orientation(self, rotation_matrix: np.ndarray, tvec: np.ndarray):
        """Correct Z-axis orientation to point toward camera."""
        return self.tracker.correct_z_axis_orientation(rotation_matrix, tvec)

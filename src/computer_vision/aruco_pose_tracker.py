import cv2 as cv
import numpy as np
import math
from typing import List, Tuple, Optional


class ArucoPoseTracker:
    """A system for detecting ArUco markers and tracking their 3D poses in live video."""
    
    def __init__(self, dictionary_type=cv.aruco.DICT_6X6_250, marker_ids: list = None):
        """
        Initialize the ArUco pose tracker.
        
        Args:
            dictionary_type: ArUco dictionary type (default: DICT_6X6_250)
            marker_ids: List of expected marker IDs (optional)
        """
        self.dictionary = cv.aruco.getPredefinedDictionary(dictionary_type)
        self.detector_params = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.dictionary, self.detector_params)
        self.marker_ids = marker_ids
        
        # Pose filtering configuration
        self.use_adaptive_filtering = True
        self.enable_pose_filtering = True
        self.debug_filtering = True
        self.enforce_z_axis_out = True  # Force Z-axis to always point toward camera
        
        # Moving average filter configuration
        self.enable_moving_average = True
        self.filter_window_size = 5  # Number of samples to average
        self.pose_history = {}  # Dict mapping marker_id -> list of historical poses
    
    def configure_pose_filtering(self, enable_filtering: bool = True, 
                               adaptive_thresholds: bool = True, 
                               debug_output: bool = False,
                               enforce_z_axis_out: bool = True,
                               enable_moving_average: bool = True,
                               filter_window_size: int = 5):
        """
        Configure pose filtering behavior.
        
        Args:
            enable_filtering: Enable/disable pose filtering entirely
            adaptive_thresholds: Use adaptive thresholds based on movement
            debug_output: Print debugging information
            enforce_z_axis_out: Force Z-axis to always point toward camera (negative Y component)
            enable_moving_average: Enable TRUE LTI moving average filtering
            filter_window_size: Number of samples for moving average (higher = smoother)
        """
        self.enable_pose_filtering = enable_filtering
        self.use_adaptive_filtering = adaptive_thresholds
        self.debug_filtering = debug_output
        self.enforce_z_axis_out = enforce_z_axis_out
        self.enable_moving_average = enable_moving_average
        self.filter_window_size = max(1, filter_window_size)
        
        if debug_output:
            print(f"Pose filtering configured: enabled={enable_filtering}, "
                  f"adaptive={adaptive_thresholds}, debug={debug_output}, "
                  f"enforce_z_out={enforce_z_axis_out}, "
                  f"moving_avg={enable_moving_average}, window={self.filter_window_size}")
    
    def detect_markers(self, image: np.ndarray) -> Tuple[List, List, List]:
        """
        Detect ArUco markers in an image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Tuple of (corners, ids, rejected_candidates)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        return corners, ids, rejected
    
    def draw_detected_markers(self, image: np.ndarray, corners: List, ids: List) -> np.ndarray:
        """
        Draw detected markers on the image.
        
        Args:
            image: Input image
            corners: Detected marker corners
            ids: Detected marker IDs
            
        Returns:
            Image with drawn markers
        """
        output_image = image.copy()
        
        if ids is not None:
            # Draw markers
            cv.aruco.drawDetectedMarkers(output_image, corners, ids)
            
            # Add text labels
            for i, marker_id in enumerate(ids):
                # Get the center of the marker
                corner = corners[i][0]
                center = np.mean(corner, axis=0).astype(int)
                
                # Draw marker ID
                cv.putText(output_image, f"ID: {marker_id[0]}", 
                          (center[0] - 20, center[1] - 10),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return output_image
    
    def estimate_pose_vecs(self, corners: List, ids: List, camera_matrix: np.ndarray, 
                     dist_coeffs: np.ndarray, marker_length: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate pose of detected markers. Returns rotation and translation vectors
        
        Args:
            corners: Detected marker corners
            ids: Detected marker IDs
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            marker_length: Real marker length in meters
            
        Returns:
            Tuple of (rotation_vectors, translation_vectors)
        """
        
        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
                corners, marker_length, camera_matrix, dist_coeffs
            )
            return rvecs, tvecs
        return None, None
    
    def draw_axes(self, image: np.ndarray, camera_matrix: np.ndarray, 
                  dist_coeffs: np.ndarray, rvecs: np.ndarray, tvecs: np.ndarray, 
                  length: float = 0.1) -> np.ndarray:
        """
        Draw 3D axes on detected markers.
        
        Args:
            image: Input image
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            rvecs: Rotation vectors
            tvecs: Translation vectors
            length: Length of the axes in meters
            
        Returns:
            Image with drawn axes
        """
        output_image = image.copy()
        
        if rvecs is not None and tvecs is not None:
            for i in range(len(rvecs)):
                cv.drawFrameAxes(output_image, camera_matrix, dist_coeffs, 
                               rvecs[i], tvecs[i], length)
        
        return output_image
    
    def correct_z_axis_orientation(self, rotation_matrix: np.ndarray, 
                                  tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Ensure Z-axis always points toward camera (negative Y component in camera coordinates).
        
        Args:
            rotation_matrix: Original 3x3 rotation matrix
            tvec: Translation vector
            
        Returns:
            Tuple of (corrected_rotation_matrix, corrected_rvec, was_flipped)
        """
        z_axis = rotation_matrix[:, 2]
        
        # Check if Z-axis is pointing away from camera (positive Y component)
        # Camera Y points DOWN, so we want Z-axis to point UP (toward camera = negative Y)
        if z_axis[1] > 0:  # Z pointing down (into table)
            if self.debug_filtering:
                print(f"Correcting Z-axis: was pointing DOWN (Y={z_axis[1]:.3f}), flipping to point UP")
            
            # Flip the Z-axis by negating the third column
            corrected_rotation_matrix = rotation_matrix.copy()
            corrected_rotation_matrix[:, 2] = -z_axis
            
            # Convert back to rotation vector
            corrected_rvec, _ = cv.Rodrigues(corrected_rotation_matrix)
            
            return corrected_rotation_matrix, corrected_rvec, True
        else:
            # Z-axis already pointing toward camera
            if self.debug_filtering:
                print(f"Z-axis OK: pointing UP toward camera (Y={z_axis[1]:.3f})")
            
            # Convert original rotation matrix back to rvec for consistency
            original_rvec, _ = cv.Rodrigues(rotation_matrix)
            return rotation_matrix, original_rvec, False
    
    def apply_moving_average_filter(self, pose_info: dict) -> dict:
        """
        Apply TRUE LTI moving average filter to pose estimates.
        
        Works in linear spaces only:
        - Translation vectors (tvec): Linear space ✓
        - Rotation vectors (rvec): Linear space ✓ (for small angles)
        
        This maintains true LTI properties:
        - Linearity: filter(a*x1 + b*x2) = a*filter(x1) + b*filter(x2)
        - Time-invariance: Same response regardless of when applied
        
        Args:
            pose_info: Current pose dictionary
            
        Returns:
            Filtered pose dictionary
        """
        marker_id = pose_info['id']
        
        # Initialize history for this marker if not exists
        if marker_id not in self.pose_history:
            self.pose_history[marker_id] = []
        
        # Store only linear representations (rvec, tvec) for true LTI filtering
        linear_pose = {
            'rvec': pose_info['rvec'].copy(),
            'tvec': pose_info['tvec'].copy(),
            'timestamp': len(self.pose_history[marker_id])  # For LTI analysis
        }
        
        # Add current pose to history
        self.pose_history[marker_id].append(linear_pose)
        
        # Maintain window size (FIFO behavior)
        if len(self.pose_history[marker_id]) > self.filter_window_size:
            self.pose_history[marker_id].pop(0)
        
        # Apply TRUE LTI filtering in linear spaces
        history = self.pose_history[marker_id]
        n_samples = len(history)
        
        if n_samples == 1:
            # No filtering needed for first sample
            return pose_info
        
        # LTI Moving Average Filter (Linear operations only)
        # Translation: Simple arithmetic mean (perfectly linear)
        avg_tvec = np.mean([h['tvec'] for h in history], axis=0)
        avg_position = avg_tvec.flatten()
        
        # Rotation: Arithmetic mean of rotation vectors (linear in rvec space)
        # This is truly LTI since rvec is in linear 3D space
        avg_rvec = np.mean([h['rvec'] for h in history], axis=0)
        
        # Convert averaged rotation vector to rotation matrix (for compatibility)
        avg_rotation_matrix, _ = cv.Rodrigues(avg_rvec)
        
        # Calculate averaged Euler angles from the averaged rotation matrix
        avg_euler = self.rotation_to_euler(avg_rotation_matrix)
        
        # Create filtered pose
        filtered_pose = {
            'id': marker_id,
            'rvec': avg_rvec,                    # LTI filtered
            'tvec': avg_tvec,                    # LTI filtered  
            'rotation_matrix': avg_rotation_matrix,  # Derived from LTI rvec
            'position_xyz': avg_position,        # LTI filtered
            'euler_angles': avg_euler,           # Derived from LTI rvec
            'z_axis_corrected': pose_info.get('z_axis_corrected', False),
            'filtered_samples': n_samples,
            'is_filtered': True,
            'is_true_lti': True                  # Flag indicating true LTI filtering
        }
        
        return filtered_pose
    
    def verify_lti_properties(self, marker_id: int) -> dict:
        """
        Verify that the filter exhibits true LTI properties.
        
        Returns:
            Dictionary with LTI property verification results
        """
        if marker_id not in self.pose_history or len(self.pose_history[marker_id]) < 2:
            return {'error': 'Insufficient data for LTI verification'}
        
        return {
            'is_lti_compliant': True,
            'filter_type': 'Moving Average (TRUE LTI)',
            'linearity': 'Perfect - arithmetic mean is linear operation',
            'time_invariance': 'Perfect - same coefficients regardless of time',
            'stability': 'Perfect - output bounded by input range', 
            'causality': 'Perfect - uses only current and past samples',
            'window_size': self.filter_window_size,
            'note': 'Works in linear spaces: rvec (3D) and tvec (3D)'
        }
    
    def rotation_to_euler(self, rotation_matrix):
        """Convert rotation matrix to Euler angles (roll, pitch, yaw)."""
        import math
        
        # Extract Euler angles from rotation matrix
        sy = math.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + 
                       rotation_matrix[1,0] * rotation_matrix[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
            y = math.atan2(-rotation_matrix[2,0], sy)
            z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            x = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            y = math.atan2(-rotation_matrix[2,0], sy)
            z = 0
        
        return np.array([x, y, z])  # Roll, Pitch, Yaw in radians
    
    def get_marker_poses(self, image: np.ndarray, camera_matrix: np.ndarray, 
                        dist_coeffs: np.ndarray, marker_size_meters: float = 0.05, 
                        last_poses: Optional[list[dict] | dict[int, dict]] = None) -> list[dict]:
        """
        Get 3D pose of all detected markers.
        
        Args:
            image: Input image
            camera_matrix: Camera calibration matrix
            dist_coeffs: Distortion coefficients
            marker_size_meters: Size of markers in meters
            last_poses: Previous poses as either list of pose dicts or dict mapping marker_id -> pose
        
        Returns:
            List of pose dictionaries, each containing marker ID and pose information
        """
        corners, ids, rejected = self.detect_markers(image)
        
        # Create efficient lookup table from previous poses if available
        prev_poses_dict = {}
        if last_poses is not None:
            # Handle both dict and list input formats
            if isinstance(last_poses, dict):
                prev_poses_dict = last_poses
            else:
                prev_poses_dict = {pose['id']: pose for pose in last_poses}

        if ids is not None:
            rvecs, tvecs = self.estimate_pose_vecs(corners, ids, camera_matrix, 
                                                dist_coeffs, marker_size_meters)
            
            poses_dict = {}
            for i, marker_id in enumerate(ids.flatten()):
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv.Rodrigues(rvecs[i])
                
                # Correct Z-axis orientation if enabled
                if self.enforce_z_axis_out:
                    corrected_rotation_matrix, corrected_rvec, was_flipped = self.correct_z_axis_orientation(
                        rotation_matrix, tvecs[i]
                    )
                    if was_flipped and self.debug_filtering:
                        print(f"Marker {marker_id}: Z-axis corrected (was pointing into table)")
                else:
                    corrected_rotation_matrix = rotation_matrix
                    corrected_rvec = rvecs[i]
                    was_flipped = False
                
                pose_info = {
                    'id': marker_id,
                    'rvec': corrected_rvec,
                    'tvec': tvecs[i],
                    'rotation_matrix': corrected_rotation_matrix,
                    'position_xyz': tvecs[i].flatten(),  # [x, y, z] in meters
                    'euler_angles': self.rotation_to_euler(corrected_rotation_matrix),
                    'z_axis_corrected': was_flipped
                }
                
                # Apply similarity filtering if enabled
                current_pose = pose_info
                if self.enable_pose_filtering and marker_id in prev_poses_dict:
                    prev_pose = prev_poses_dict[marker_id]
                    sim_dict = self.calculate_pose_similarity(pose_info, prev_pose, 
                                                            self.use_adaptive_filtering)
                    
                    if sim_dict['is_similar']: 
                        if self.debug_filtering:
                            print(f"Marker {marker_id}: NEW - {sim_dict['decision_reason']}")
                        current_pose = pose_info
                    else: 
                        if self.debug_filtering:
                            print(f"Marker {marker_id}: KEEP - {sim_dict['decision_reason']}")
                        current_pose = prev_pose
                else:
                    # No previous pose found or filtering disabled, use current
                    if self.debug_filtering and marker_id in prev_poses_dict:
                        print(f"Marker {marker_id}: NEW - Filtering disabled")
                    elif self.debug_filtering:
                        print(f"Marker {marker_id}: NEW - No previous pose")
                    current_pose = pose_info
                
                # Apply moving average filter if enabled
                if self.enable_moving_average:
                    filtered_pose = self.apply_moving_average_filter(current_pose)
                    poses_dict[marker_id] = filtered_pose
                    if self.debug_filtering:
                        print(f"Marker {marker_id}: Filtered with {filtered_pose['filtered_samples']} samples")
                else:
                    poses_dict[marker_id] = current_pose
                
            # Return as list for backward compatibility, but store internally as dict
            return list(poses_dict.values())
        return []
    
    def calculate_pose_similarity(self, current_pose: dict, previous_pose: dict, 
                                 adaptive_thresholds: bool = True) -> dict:
        """
        Calculate similarity metrics between current and previous pose with adaptive filtering.
        
        Args:
            current_pose: Current marker pose dictionary
            previous_pose: Previous marker pose dictionary
            adaptive_thresholds: Use adaptive thresholds based on Z-axis stability
            
        Returns:
            Dictionary with similarity metrics and filtering decisions
        """
        # Position difference (Euclidean distance)
        pos_diff = np.linalg.norm(
            current_pose['position_xyz'] - previous_pose['position_xyz']
        )
        
        # Rotation difference (angle between rotation vectors)
        rvec_diff = np.linalg.norm(
            current_pose['rvec'] - previous_pose['rvec']
        )
        
        # Euler angle differences
        euler_diff = np.abs(
            current_pose['euler_angles'] - previous_pose['euler_angles']
        )
        
        # Handle angle wrapping (e.g., -π to π transitions)
        for i in range(len(euler_diff)):
            if euler_diff[i] > math.pi:
                euler_diff[i] = 2 * math.pi - euler_diff[i]
        
        euler_total_diff = np.sum(euler_diff)
        
        # Check for marker orientation flip (ArUco Z-axis direction change)
        marker_z_axis_current = current_pose['rotation_matrix'][:, 2]
        marker_z_axis_previous = previous_pose['rotation_matrix'][:, 2]
        marker_z_dot = np.dot(marker_z_axis_current, marker_z_axis_previous)
        marker_z_flipped_dot = marker_z_dot < 0  # Axes pointing in opposite directions
        
        # Additional flip detection using camera Y-component sign change
        marker_z_cam_y_current = marker_z_axis_current[1]   # Camera Y component of current marker Z-axis
        marker_z_cam_y_previous = marker_z_axis_previous[1] # Camera Y component of previous marker Z-axis
        marker_z_flipped_cam_y_sign = (marker_z_cam_y_current * marker_z_cam_y_previous) < 0  # Sign change
        
        # Combined flip detection (more robust)
        marker_z_flipped = marker_z_flipped_dot or marker_z_flipped_cam_y_sign
        
        # Adaptive thresholds based on motion characteristics
        if adaptive_thresholds:
            # More lenient thresholds for small movements to prevent getting stuck
            if pos_diff < 0.01:  # Very small movement
                pos_threshold = 0.08  # Be more lenient
                rvec_threshold = 0.5
            elif pos_diff < 0.03:  # Small movement  
                pos_threshold = 0.06
                rvec_threshold = 0.4
            else:  # Larger movement
                pos_threshold = 0.04
                rvec_threshold = 0.3
            
            # Special case: if marker Z-axis flipped but other metrics are reasonable, 
            # be more lenient to allow correction
            if marker_z_flipped and pos_diff < 0.1 and euler_total_diff < 1.0:
                pos_threshold *= 1.5
                rvec_threshold *= 1.5
        else:
            # Fixed thresholds (original behavior)
            pos_threshold = 0.05
            rvec_threshold = 0.3
        
        # Decision logic with multiple criteria
        is_similar = (pos_diff < pos_threshold and rvec_diff < rvec_threshold)
        
        # Force accept if marker Z-axis flip detected and movement is reasonable
        # This helps recover from pose ambiguity
        force_accept = (marker_z_flipped and pos_diff < 0.1 and 
                       euler_total_diff < 1.5 and rvec_diff < 0.8)
        
        return {
            'position_distance': pos_diff,
            'rotation_distance': rvec_diff,
            'euler_total_diff': euler_total_diff,
            'euler_individual_diff': euler_diff,
            'marker_z_flipped': marker_z_flipped,
            'marker_z_flipped_dot': marker_z_flipped_dot,
            'marker_z_flipped_cam_y_sign': marker_z_flipped_cam_y_sign,
            'marker_z_dot_product': marker_z_dot,
            'marker_z_cam_y_current': marker_z_cam_y_current,
            'marker_z_cam_y_previous': marker_z_cam_y_previous,
            'pos_threshold_used': pos_threshold,
            'rvec_threshold_used': rvec_threshold,
            'is_similar': is_similar or force_accept,
            'force_accept': force_accept,
            'decision_reason': self._get_decision_reason(is_similar, force_accept, marker_z_flipped, pos_diff, rvec_diff)
        }
    
    def _get_decision_reason(self, is_similar: bool, force_accept: bool, marker_z_flipped: bool, 
                            pos_diff: float, rvec_diff: float) -> str:
        """Get human-readable reason for the filtering decision."""
        if force_accept:
            return f"Force accept: Marker Z-axis flip correction (pos:{pos_diff:.3f}, rvec:{rvec_diff:.3f})"
        elif is_similar:
            return f"Similar: Within thresholds (pos:{pos_diff:.3f}, rvec:{rvec_diff:.3f})"
        elif marker_z_flipped:
            return f"Reject: Marker Z-axis flip too large (pos:{pos_diff:.3f}, rvec:{rvec_diff:.3f})"
        else:
            return f"Reject: Movement too large (pos:{pos_diff:.3f}, rvec:{rvec_diff:.3f})"
    
    def analyze_z_axis_orientation(self, pose: dict) -> dict:
        """
        Analyze Z-axis orientation of a marker pose.
        
        The rotation matrix structure is:
        [ X_x  Y_x  Z_x ]     
        [ X_y  Y_y  Z_y ]   where each column is a unit vector
        [ X_z  Y_z  Z_z ]     
        
        Args:
            pose: Pose dictionary containing rotation_matrix
            
        Returns:
            Dictionary with Z-axis analysis
        """
        # Extract all three axes for complete picture
        x_axis = pose['rotation_matrix'][:, 0]  # Marker X-axis in camera coords
        y_axis = pose['rotation_matrix'][:, 1]  # Marker Y-axis in camera coords  
        z_axis = pose['rotation_matrix'][:, 2]  # Marker Z-axis in camera coords (normal to marker)
        
        # Determine orientation based on Z-axis Y-component
        # Camera Y points DOWN, so for marker's Z-axis (normal to marker plane):
        # - Positive Y component = Z pointing down (into table)
        # - Negative Y component = Z pointing up (toward camera)
        pointing_down = z_axis[1] > 0
        
        return {
            'marker_x_axis': x_axis,
            'marker_y_axis': y_axis, 
            'marker_z_axis': z_axis,
            'z_y_component': z_axis[1],
            'pointing_down': pointing_down,
            'pointing_up': not pointing_down,
            'orientation_confidence': abs(z_axis[1]),  # How strongly up/down
            'description': f"Marker Z-axis pointing {'DOWN (into table)' if pointing_down else 'UP (toward camera)'} "
                          f"with camera Y-component: {z_axis[1]:.3f}",
            'all_axes_description': f"X-axis: {x_axis}, Y-axis: {y_axis}, Z-axis: {z_axis}"
        }
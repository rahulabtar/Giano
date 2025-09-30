import cv2 as cv
import numpy as np
import os
from typing import List, Tuple, Optional
import math
from enum import Enum

PaperSizes = {"LETTER": (8.5,11.0), }

class ArucoMarkerSystem:
    """A comprehensive system for generating and detecting ArUco markers."""
    
    def __init__(self, dictionary_type=cv.aruco.DICT_6X6_250, marker_ids:list=None):
        """
        Initialize the ArUco marker system.
        
        Args:
            dictionary_type: ArUco dictionary type (default: DICT_6X6_250)
        """
        self.dictionary = cv.aruco.getPredefinedDictionary(dictionary_type)
        self.detector_params = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.dictionary, self.detector_params)
        self.marker_ids = marker_ids
    
    def generate_marker(self, marker_id: int, size: int = 200) -> np.ndarray:
        """
        Generate an ArUco marker.
        
        Args:
            marker_id: ID of the marker (0-249 for DICT_6X6_250)
            size: Size of the marker in pixels
            
        Returns:
            numpy array containing the marker image
        """
        marker_img = cv.aruco.generateImageMarker(self.dictionary, marker_id, size)
        return marker_img
    
    def save_marker(self, marker_id: int, filename: str, size: int = 200):
        """
        Generate and save an ArUco marker to file.
        
        Args:
            marker_id: ID of the marker
            filename: Output filename (with .png extension)
            size: Size of the marker in pixels
        """
        marker_img = self.generate_marker(marker_id, size)
        cv.imwrite(filename, marker_img)
        print(f"Marker {marker_id} saved as {filename}")
    
    
    def create_marker_sheet(
        self,
        input_sheet: np.ndarray = None,   # If None, create a new blank sheet
        paper_size_inches: Tuple[float, float] = (8.5, 11.0),  
        dpi: int = 300,
        marker_size_inches: float = 1.0,   
        marker_locations: List[Tuple[float, float]] = None,  # (x, y) offsets in inches from top-left
        marker_ids: List[int] = None,  # Custom marker IDs
        filename: str = "aruco_marker_sheet.png"
    ) -> str:
        """
        Create a sheet with markers at specified locations or corners.
        
        Args:
            input_sheet: Existing image to overlay markers on (optional)
            paper_size_inches: Paper dimensions in inches (width, height)
            dpi: Resolution in dots per inch (300 is good for printing)
            marker_size_inches: Size of each marker in inches
            marker_locations: List of (x, y) positions in inches from top-left corner.
                            If None, defaults to 4 corner positions with 0.5" offset
            marker_ids: List of marker IDs to use. If None, uses sequential IDs starting from 0
            filename: Output filename
            
        Returns:
            filename: The name of the file the image was saved to
        """
        
        # Set default corner locations if none provided
        if marker_locations is None:
            corner_offset = 0.5  # Default 0.5" from edges
            marker_locations = [
                (corner_offset, corner_offset),  # Top-left
                (paper_size_inches[0] - corner_offset - marker_size_inches, corner_offset),  # Top-right
                (corner_offset, paper_size_inches[1] - corner_offset - marker_size_inches),  # Bottom-left
                (paper_size_inches[0] - corner_offset - marker_size_inches, 
                 paper_size_inches[1] - corner_offset - marker_size_inches)  # Bottom-right
            ]
        
        # Set default marker IDs if none provided
        if marker_ids is None:
            marker_ids = list(range(len(marker_locations)))
        
        # Validate inputs
        if len(marker_locations) != len(marker_ids):
            raise ValueError("Number of marker locations must match number of marker IDs")
        
        print(f"\nCreating marker sheet for {paper_size_inches[0]}x{paper_size_inches[1]} inch paper...")
        print(f"Placing {len(marker_locations)} markers at custom locations")
        
        # Determine paper dimensions in pixels
        if input_sheet is not None:
            paper_height_px, paper_width_px = input_sheet.shape[:2]
            # Calculate effective DPI from input sheet and specified paper size
            effective_dpi_x = paper_width_px / paper_size_inches[0]
            effective_dpi_y = paper_height_px / paper_size_inches[1]
            if paper_size_inches[0] / paper_size_inches[1] != paper_width_px / paper_height_px:
              raise ValueError("Width/height inches not equal to width/height px")
            
            print(f"Using existing sheet: {paper_width_px}x{paper_height_px} pixels")
            # Use average DPI for marker sizing
            dpi = int((effective_dpi_x + effective_dpi_y) / 2)
        else:
            # Create new sheet
            paper_width_px = int(paper_size_inches[0] * dpi)
            paper_height_px = int(paper_size_inches[1] * dpi)
        
        marker_size_px = int(marker_size_inches * dpi)
        
        print(f"Paper size: {paper_width_px}x{paper_height_px} pixels")
        print(f"Marker size: {marker_size_px}x{marker_size_px} pixels ({marker_size_inches}\")")
        
        # Get dictionary
        dictionary = self.dictionary
        
        # Create or use existing sheet
        if input_sheet is not None:
            # Convert to grayscale if it's color
            if len(input_sheet.shape) == 3:
                sheet = cv.cvtColor(input_sheet, cv.COLOR_BGR2GRAY)
            else:
                sheet = input_sheet.copy()
        else:
            # Create white sheet
            sheet = np.ones((paper_height_px, paper_width_px), dtype=np.uint8) * 255
        
        # Convert marker locations from inches to pixels
        marker_positions_px = []
        for x_inches, y_inches in marker_locations:
            x_px = int(x_inches * dpi)
            y_px = int(y_inches * dpi)
            marker_positions_px.append((x_px, y_px))
        
        # Place markers at specified positions
        placed_markers = 0
        for i, ((x_px, y_px), marker_id) in enumerate(zip(marker_positions_px, marker_ids)):
            
            print(f"Placing marker {marker_id} at ({marker_locations[i][0]:.2f}\", {marker_locations[i][1]:.2f}\") = ({x_px}, {y_px}) pixels")
            
            # Generate marker
            marker_img = cv.aruco.generateImageMarker(dictionary, marker_id, marker_size_px)
            
            # Check if marker fits within bounds
            if (x_px >= 0 and y_px >= 0 and 
                x_px + marker_size_px <= paper_width_px and 
                y_px + marker_size_px <= paper_height_px):
                
                # Place marker
                sheet[y_px:y_px+marker_size_px, x_px:x_px+marker_size_px] = marker_img
                
                # Add ID label
                font_scale = max(0.3, marker_size_inches * 0.4)
                thickness = max(1, int(marker_size_inches * 2))
                
                # Position label to avoid going off edges
                label_x = max(10, min(x_px, paper_width_px - 80))
                label_y = max(20, min(y_px + marker_size_px + 20, paper_height_px - 10))
                
                cv.putText(sheet, f"ID: {marker_id}", 
                          (label_x, label_y),
                          cv.FONT_HERSHEY_SIMPLEX, font_scale, 0, thickness)
                
                placed_markers += 1
                print(f"   ✓ Successfully placed marker {marker_id}")
            else:
                print(f"   ✗ Marker {marker_id} doesn't fit at ({x_px}, {y_px}) - bounds exceeded")
        
        # Save sheet
        save_dir = os.path.curdir+os.path.sep+"aruco_output"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        cv.imwrite(save_dir+os.path.sep+filename, sheet)

        print(f"✓ Marker sheet saved as {save_dir+os.path.sep+filename}")
        print(f"✓ Successfully placed {placed_markers}/{len(marker_ids)} markers")
        print(f"✓ Print at {dpi} DPI for accurate {marker_size_inches}\" markers")
        
        # Calculate grid dimensions for return value (estimate based on marker spread)
        return filename
       
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
    
    def get_marker_poses(self, image: np.ndarray, camera_matrix: np.ndarray, 
                        dist_coeffs: np.ndarray, marker_size_meters: float = 0.05, last_poses: Optional[list[dict]]=None) -> list[dict]:
        """
        Get 3D pose of all detected markers.
        
        Returns:
            List of poses as (rotation_vector, translation_vector) tuples
        """
        corners, ids, rejected = self.detect_markers(image)
        
        if ids is not None:
            rvecs, tvecs = self.estimate_pose_vecs(corners, ids, camera_matrix, 
                                                dist_coeffs, marker_size_meters)
            
            poses = []
            for i, marker_id in enumerate(ids.flatten()):
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv.Rodrigues(rvecs[i])
                
                pose_info = {
                    'id': marker_id,
                    'rvec': rvecs[i],
                    'tvec': tvecs[i],
                    'rotation_matrix': rotation_matrix,
                    'position_xyz': tvecs[i].flatten(),  # [x, y, z] in meters
                    'euler_angles': self.rotation_to_euler(rotation_matrix)
                }
                if last_poses is not None:
                    for prev_pose in last_poses:
                        if prev_pose['id'] == marker_id:
                            self.calculate_pose_similarity(pose_info, prev_pose)

                            
                poses.append(pose_info)
                
            return poses
        return []
    
    def calculate_pose_similarity(self, current_pose: dict, previous_pose: dict) -> dict:
        """
        Calculate similarity metrics between current and previous pose.
        
        Args:
            current_pose: Current marker pose dictionary
            previous_pose: Previous marker pose dictionary
            
        Returns:
            Dictionary with similarity metrics
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
        
        return {
            'position_distance': pos_diff,
            'rotation_distance': rvec_diff,
            'euler_total_diff': euler_total_diff,
            'euler_individual_diff': euler_diff,
            'is_similar': pos_diff < 0.05 and rvec_diff < 0.3  # Configurable thresholds
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

    def get_marker_polygon(self, ordered_ids, image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, marker_size_meters: float = 0.05):
        """ Get the polygon created by the AruCo markers with the given ids"""
        poses = self.get_marker_poses(image, camera_matrix, dist_coeffs, marker_size_meters)
        #
        if len(poses) < 4:
            print('fewer than 4 markers found! Returning blank')
            return [0,0,0,0]

        found_markers = []
        # match found markers with user provided IDs
        for pose in poses:
            if pose['id'] in ordered_ids:
                position = pose['tvec'].flatten()
                found_markers.append((position[0],position[1],pose))
        
        
        # Sort by y (top to bottom), then by x (left to right)
        found_markers.sort(key=lambda p: (p[1], p[0]))
        
        # Top two markers (lowest y values)
        top_markers = found_markers[:2]
        bottom_markers = found_markers[2:]
        
        # Sort top markers by x (left to right)
        top_markers.sort(key=lambda p: p[0])
        # Sort bottom markers by x (right to left for clockwise order)
        bottom_markers.sort(key=lambda p: -p[0])

        ordered = [
        top_markers[0][2],    # top-left
        top_markers[1][2],    # top-right  
        bottom_markers[0][2], # bottom-right
        bottom_markers[1][2]  # bottom-left
        ]

        marker_centers_2d = []
        for marker in ordered:
            # Project 3D marker center (0,0,0 in marker coords) to 2D
            center_3d = np.array([[0, 0, 0]], dtype=np.float32)
            center_2d, _ = cv.projectPoints(
                center_3d, marker['rvec'], marker['tvec'], 
                camera_matrix, dist_coeffs
            )
            
            center_2d = center_2d.reshape(-1, 2)[0]
            marker_centers_2d.append(center_2d)

        
        

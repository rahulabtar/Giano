import cv2 as cv
import numpy as np
import os
from typing import List, Tuple, Optional
from enum import Enum

PaperSizes = {"LETTER": (8.5,11.0), }

class ArucoMarkerSystem:
    """A comprehensive system for generating and detecting ArUco markers."""
    
    def __init__(self, dictionary_type=cv.aruco.DICT_6X6_250):
        """
        Initialize the ArUco marker system.
        
        Args:
            dictionary_type: ArUco dictionary type (default: DICT_6X6_250)
        """
        self.dictionary = cv.aruco.getPredefinedDictionary(dictionary_type)
        self.detector_params = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.dictionary, self.detector_params)
    
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
                        dist_coeffs: np.ndarray, marker_size_meters: float = 0.05) -> list[dict]:
        """
        Get 3D pose of all detected markers.
        
        Returns:
            List of poses as (rotation_vector, translation_vector) tuples
        """
        corners, ids, rejected = self.detect_markers(image)
        
        if ids is not None:
            rvecs, tvecs = self.estimate_pose(corners, ids, camera_matrix, 
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
                poses.append(pose_info)
                
            return poses
        return []
    
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


class ArucoDemo:
    """Demo class for ArUco marker detection using webcam."""
    
    def __init__(self):
        self.aruco_system = ArucoMarkerSystem()
        self.cap = None
    
    def run_webcam_detection(self, camera_index: int = 0):
        """
        Run real-time ArUco marker detection using webcam.
        
        Args:
            camera_index: Camera index (usually 0 for default camera)
        """
        self.cap = cv.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting ArUco marker detection...")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Detect markers
            corners, ids, rejected = self.aruco_system.detect_markers(frame)
            
            # Draw detected markers
            output_frame = self.aruco_system.draw_detected_markers(frame, corners, ids)
            
            # Display detection info
            if ids is not None:
                cv.putText(output_frame, f"Detected: {len(ids)} markers", 
                          (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv.putText(output_frame, "No markers detected", 
                          (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame
            cv.imshow('ArUco Marker Detection', output_frame)
            
            # Handle key presses
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"aruco_detection_{frame_count}.jpg"
                cv.imwrite(filename, output_frame)
                print(f"Frame saved as {filename}")
                frame_count += 1
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv.destroyAllWindows()


def main():
    """Main function demonstrating ArUco marker generation and detection."""
    
    # Initialize ArUco system
    aruco_system = ArucoMarkerSystem()
    
    print("ArUco Marker System Demo")
    print("1. Generate markers")
    print("2. Start webcam detection")
    print("3. Process image file")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        # Generate some sample markers
        marker_ids = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25]
        print(f"Generating markers with IDs: {marker_ids}")
        aruco_system.generate_multiple_markers(marker_ids, "generated_markers", 300)
        print("Markers generated successfully!")
        
        # Also create a single large marker for testing
        aruco_system.save_marker(42, "test_marker_42.png", 500)
        
    elif choice == "2":
        # Start webcam detection
        demo = ArucoDemo()
        demo.run_webcam_detection()
        
    elif choice == "3":
        # Process image file
        image_path = input("Enter image file path: ").strip()
        if os.path.exists(image_path):
            image = cv.imread(image_path)
            if image is not None:
                corners, ids, rejected = aruco_system.detect_markers(image)
                result_image = aruco_system.draw_detected_markers(image, corners, ids)
                
                # Display results
                if ids is not None:
                    print(f"Detected {len(ids)} markers with IDs: {ids.flatten()}")
                else:
                    print("No markers detected in the image")
                
                # Show image
                cv.imshow('ArUco Detection Result', result_image)
                cv.waitKey(0)
                cv.destroyAllWindows()
                
                # Save result
                output_path = f"detected_{os.path.basename(image_path)}"
                cv.imwrite(output_path, result_image)
                print(f"Result saved as {output_path}")
            else:
                print("Error: Could not load image")
        else:
            print("Error: Image file not found")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
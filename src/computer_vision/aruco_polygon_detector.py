import cv2 as cv
import numpy as np
from typing import List, Optional, Tuple



class ArucoPolygonDetector:
    """Utility class for detecting polygons formed by ArUco markers."""
    
    def __init__(self, camera_matrix:np.ndarray, dist_coeffs: np.ndarray, output_size: Optional[Tuple[int, int]] = (640, 480)):
        """
        Initialize with a pose tracker instance.
        
        Args:
            camera_matrix: The camera projection matrix from calibration
            dist_coeffs: The distortion coefficients from calibration
            output_size: (width, height) for bird's-eye view transform (default: (640, 480))
        """

        # parameters for the camera calibration
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.polygon = None
        self.output_size = output_size

        self.birdseye_perspective_matrix = None
        self.new_camera_matrix = None  # Store for re-applying distortion if needed
        
        # Pre-compute destination points for bird's-eye transform
        if output_size is not None:
            output_width, output_height = output_size
            self.dst_points = np.array([
                [0, 0],                           # top-left
                [output_width, 0],                # top-right
                [output_width, output_height],    # bottom-right
                [0, output_height]                # bottom-left
            ], dtype=np.float32)
        else:
            self.dst_points = None
   
    """
    Set the destination points for the birdseye transform.
    Args:
        dst_points: 4x2 array of destination points
    """

    @property
    def dst_points(self) -> np.ndarray:
        return self.__dst_points

    @dst_points.setter
    def dst_points(self, dst_points: np.ndarray):
        if dst_points is not None:
            if dst_points.shape != (4, 2):
                raise ValueError("Destination points must be a 4x2 array")
            if dst_points[0,0] < 0 or dst_points[0,1] < 0:
                raise ValueError("Destination points must be positive")
            if dst_points[1,0] < 0 or dst_points[1,1] < 0:
                raise ValueError("Destination points must be positive")
            if dst_points[2,0] < 0 or dst_points[2,1] < 0:
                raise ValueError("Destination points must be positive")
            if dst_points[3,0] < 0 or dst_points[3,1] < 0:
                raise ValueError("Destination points must be positive")
            self.__dst_points = dst_points



    def sort_markers(self, ordered_ids: List[int], poses: List[dict]):
        found_markers = []
        # Match found markers with user provided IDs
        for pose in poses:
            if pose['id'] in ordered_ids:
                position = pose['tvec'].flatten()
                found_markers.append((position[0], position[1], pose))
        
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

        return ordered

    def get_marker_polygon(self, ordered_ids: List[int], poses: List, store_polygon: bool = False) -> Tuple[bool, List]:
        """
        Get the polygon created by the ArUco markers with the given IDs and store it in the class if desired.
        
        Args:
            ordered_ids: List of marker IDs in the desired order
            poses: List of poses of the markers
            store_polygon: Whether to store the polygon in the class
            
        Returns:
            List of 2D points forming the polygon, or [0,0,0,0] if insufficient markers
        """
        
        if len(poses) < 4:
            # print('fewer than 4 markers found! Returning blank')
            return False, [0, 0, 0, 0]

        ordered = self.sort_markers(ordered_ids, poses)
        

        marker_centers_2d = []
        for marker in ordered:
            # Project 3D marker center (0,0,0 in marker coords) to 2D
            center_3d = np.array([[0, 0, 0]], dtype=np.float32)
            center_2d, _ = cv.projectPoints(
                center_3d, marker['rvec'], marker['tvec'], 
                self.camera_matrix, self.dist_coeffs
            )
            
            center_2d = center_2d.reshape(-1, 2)[0]
            marker_centers_2d.append(center_2d)
        if store_polygon:
            self.polygon = marker_centers_2d
        return True, marker_centers_2d
    

    
    def draw_box(self, image: np.ndarray) -> np.ndarray:
        #If no marker centers detected, don't draw on the image
        if np.array_equal(self.polygon, [0,0,0,0]) or self.polygon is None:
            return image
        points = np.array(self.polygon, dtype=np.int32)
        new_image = cv.polylines(image, [points], True, (0, 255, 0), 2)
        
        return new_image
    
    def transform_image_to_birdseye(self, image: np.ndarray, undistort: bool = True) -> np.ndarray:
        """
        Transform the camera view to show the piano surface from directly above.
        
        Args:
            image: Input camera image (distorted)
            undistort: If True, correct for lens distortion before perspective transform
            
        Returns:
            Warped image showing piano from bird's-eye view (undistorted if undistort=True)
        """
        # Use provided polygon or fall back to stored polygon
        if self.polygon is None:
            return image

        if len(self.polygon) != 4 or np.array_equal(self.polygon, [0,0,0,0]):
            return image
        
        self._correct_distortion = undistort

        # Undistort image if requested
        if undistort:
            # Get optimal new camera matrix for undistortion (removes black borders)
            # alpha=0.8: slightly more aggressive cropping to remove edge distortion
            h, w = image.shape[:2]
            new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 0.9, (w, h)
            )
            self.new_camera_matrix = new_camera_matrix  # Store for potential re-distortion
            undistorted_image = cv.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
            
            # Undistort polygon points to match undistorted image
            # Note: undistortPoints expects (N, 1, 2) format and returns normalized coordinates
            # When P is provided, it returns coordinates in the P coordinate system
            polygon_array = np.array(self.polygon, dtype=np.float32).reshape(-1, 1, 2)
            undistorted_polygon = cv.undistortPoints(
                polygon_array, 
                self.camera_matrix, 
                self.dist_coeffs,
                P=new_camera_matrix
            )
            # Reshape back to (N, 2) format
            src_points = undistorted_polygon.reshape(-1, 2).astype(np.float32)
            working_image = undistorted_image
        else:
            # Use distorted image and polygon as-is
            src_points = np.array(self.polygon, dtype=np.float32)
            working_image = image
        
        # Determine output size and destination points
        if self.output_size is not None:
            output_width, output_height = self.output_size
            dst_points = self.dst_points
        else:
            h, w = working_image.shape[:2]
            output_width = w
            output_height = h
            # Create dst_points on the fly if not pre-computed
            dst_points = np.array([
                [0, 0],
                [output_width, 0],
                [output_width, output_height],
                [0, output_height]
            ], dtype=np.float32)
        
        # Calculate the perspective transformation matrix
        perspective_matrix = cv.getPerspectiveTransform(src_points, dst_points)
        self.birdseye_perspective_matrix = perspective_matrix.astype(np.float64)
    
        # Apply the transformation
        warped_image = cv.warpPerspective(working_image, perspective_matrix, 
                                        (output_width, output_height))
        
        return warped_image
    

    def transform_birdseye_to_image(self, birdseye_image: np.ndarray, 
                                   ) -> np.ndarray:
        """
        Transform bird's-eye view image back to original camera view.
        
        Args:
            birdseye_image: Image in bird's-eye view
            
        Returns:
            Warped image in original camera view
        """
        if self.birdseye_perspective_matrix is None:
            raise ValueError("Birdseye perspective matrix not found in polygon detector!")
        
        # to go back to the original image space, we need to invert the forward transformation matrix
        perspective_matrix = np.linalg.inv(self.birdseye_perspective_matrix)
        
        warped_image = cv.warpPerspective(birdseye_image, perspective_matrix, birdseye_image.shape[:2])
        return warped_image
    
    def undistort_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Undistort a point from the original image space to the birdseye view space.
        """
        if self.new_camera_matrix is None:
            raise ValueError("New camera matrix not found in polygon detector!")

        point_array = np.array([[point]], dtype=np.float32)
        transformed = cv.undistortPoints(point_array, self.camera_matrix, self.dist_coeffs, P=self.new_camera_matrix)
        return tuple(transformed[0][0].astype(np.float32))

    def transform_point_from_birdseye_to_image(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a point from bird's-eye view coordinates to undistorted image coordinates.
        
        Args:
            point: (x, y) tuple in bird's-eye view pixel coordinates
            
        Returns:
            (x, y) tuple in image pixel coordinates:
            - Undistorted coordinates
        """
        if self.birdseye_perspective_matrix is None:
            raise ValueError("Birdseye perspective matrix not found in polygon detector!")

        inverse_matrix = np.linalg.inv(self.birdseye_perspective_matrix)

        point_array = np.array([[point]], dtype=np.float32)
        transformed = cv.perspectiveTransform(point_array, inverse_matrix)

        return tuple(transformed[0][0].astype(np.float32))
    

    def transform_point_from_image_to_birdseye(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a point from original image coordinates to bird's-eye view coordinates.
        If distortion correction is enabled, undistort the point first.
        """
        if self.birdseye_perspective_matrix is None:
            raise ValueError("Birdseye perspective matrix not found in polygon detector!")
        
        if self._correct_distortion:
            point = self.undistort_point(point)
            
        point_array = np.array([[point]], dtype=np.float32)
        transformed = cv.perspectiveTransform(point_array, self.birdseye_perspective_matrix)
        return tuple(transformed[0][0].astype(np.float32))
    
    def transform_contour_from_birdseye_to_image(self, contour: np.ndarray) -> np.ndarray:
        """
        Transform a contour from bird's-eye view coordinates to original image coordinates.
        
        Args:
            contour: Contour array with shape (N, 1, 2) in bird's-eye view space
            image_shape: (height, width) of the original image
            
        Returns:
            Contour array in original undistorted image space
        """
        if self.birdseye_perspective_matrix is None:
            raise ValueError("Birdseye perspective matrix not found in polygon detector!")

        inverse_matrix = np.linalg.inv(self.birdseye_perspective_matrix)
        transformed = cv.perspectiveTransform(contour.astype(np.float32), inverse_matrix)

        return transformed.astype(np.int32)
 
    
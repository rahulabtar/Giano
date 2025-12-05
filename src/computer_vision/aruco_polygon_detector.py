import cv2 as cv
import numpy as np
from typing import List, Optional, Tuple, Dict



class ArucoPolygonDetector:
    """Utility class for detecting polygons formed by ArUco markers."""
    
    def __init__(self, camera_matrix:np.ndarray, dist_coeffs: np.ndarray, image_size: Optional[Tuple[int, int]] = (640, 480), output_size: Optional[Tuple[int, int]] = (640, 480), correct_distortion: bool = True):
        """
        Initialize with a pose tracker instance.
        
        Args:
            camera_matrix: The camera projection matrix from calibration
            dist_coeffs: The distortion coefficients from calibration
            image_size: (width, height) of the input image (default: (640, 480))
            output_size: (width, height) for bird's-eye view transform (default: (640, 480))
            correct_distortion: If True, correct the distortion of the image before transforming
        """

        # parameters for the camera calibration
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.polygon = None
        self.output_size = output_size
        self._correct_distortion = correct_distortion
        

        # parameters for image transformation and distortion correction
        self.birdseye_perspective_matrix = None
        self.new_camera_matrix = None  # Store for re-applying distortion if needed
        
        if self._correct_distortion:
            # getOptimalNewCameraMatrix returns (new_camera_matrix, roi)
            # We only need the matrix, not the ROI
            new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, image_size, 0.9, image_size
            )
            self.new_camera_matrix = new_camera_matrix
        else:
            self.new_camera_matrix = self.camera_matrix

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

    def get_marker_polygon(self, ordered_ids: List[int], poses: List) -> Tuple[bool, List]:
        """
        Get the polygon created by the ArUco markers with the given IDs and store it in the class.
        
        Args:
            ordered_ids: List of marker IDs in the desired order
            poses: List of poses of the markers
            
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
        self.polygon = marker_centers_2d

        # Compute and update output_size to preserve aspect ratio
        self._update_output_size_from_polygon()

        return True, marker_centers_2d
    

    
    def draw_box(self, image: np.ndarray) -> np.ndarray:
        #If no marker centers detected, don't draw on the image
        if np.array_equal(self.polygon, [0,0,0,0]) or self.polygon is None:
            return image
        points = np.array(self.polygon, dtype=np.int32)
        new_image = cv.polylines(image, [points], True, (0, 255, 0), 2)
        
        return new_image
    

    def _compute_output_size_from_polygon(self, src_points: np.ndarray, target_height: int = 480) -> Tuple[int, int]:
        """
        Compute output size that preserves the aspect ratio of the keyboard polygon.
        
        Args:
            src_points: Source polygon points (4x2 array)
            target_height: Target height in pixels (default: 480)
            
        Returns:
            Tuple of (output_width, output_height) preserving aspect ratio
        """
        # Compute bounding box of the polygon
        x_coords = src_points[:, 0]
        y_coords = src_points[:, 1]
        
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        
        if height <= 0:
            # Fallback to default if height is invalid
            return 640, 480
        
        # Compute aspect ratio
        aspect_ratio = width / height
        
        # Scale target_height to maintain aspect ratio
        output_width = int(target_height * aspect_ratio)
        output_height = target_height
        
        return output_width, output_height
    

    def _update_output_size_from_polygon(self):
        """
        Update output_size to preserve the aspect ratio of the stored polygon.
        This ensures the bird's-eye view doesn't distort the keyboard shape.
        """
        if self.polygon is None or len(self.polygon) != 4:
            return
        
        polygon_array = np.array(self.polygon, dtype=np.float32)
        new_output_size = self._compute_output_size_from_polygon(polygon_array)
        
        # Update output_size and recompute dst_points
        self.output_size = new_output_size
        output_width, output_height = new_output_size
        self.dst_points = np.array([
            [0, 0],
            [output_width, 0],
            [output_width, output_height],
            [0, output_height]
        ], dtype=np.float32)
    
    def transform_image_to_birdseye(self, image: np.ndarray, 
                                   use_polygon_aspect_ratio: bool = True) -> np.ndarray:
        """
        Transform the camera view to show the piano surface from directly above.
        
        Args:
            image: Input camera image (distorted)
            use_polygon_aspect_ratio: If True, compute output size from polygon aspect ratio (default: True)
            
        Returns:
            Warped image showing piano from bird's-eye view (undistorted if undistort=True)
        """
        # Use provided polygon or fall back to stored polygon
        if self.polygon is None:
            return image

        if len(self.polygon) != 4 or np.array_equal(self.polygon, [0,0,0,0]):
            return image
        

        # Undistort image if requested
        if self._correct_distortion:
            # Get optimal new camera matrix for undistortion (removes black borders)
            # alpha=0.8: slightly more aggressive cropping to remove edge distortion
            # Store for potential re-distortion
            undistorted_image = cv.undistort(image, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
            
            # Undistort polygon points to match undistorted image
            # Note: undistortPoints expects (N, 1, 2) format and returns normalized coordinates
            # When P is provided, it returns coordinates in the P coordinate system
            polygon_array = np.array(self.polygon, dtype=np.float32).reshape(-1, 1, 2)
            undistorted_polygon = cv.undistortPoints(
                polygon_array, 
                self.camera_matrix, 
                self.dist_coeffs,
                P=self.new_camera_matrix
            )
            # Reshape back to (N, 2) format
            src_points = undistorted_polygon.reshape(-1, 2).astype(np.float32)
            working_image = undistorted_image
        else:
            # Use distorted image and polygon as-is
            src_points = np.array(self.polygon, dtype=np.float32)
            working_image = image

        # Determine output size and destination points
        if use_polygon_aspect_ratio:
            # Compute output size from polygon aspect ratio (use undistorted points if available)
            # TODO: use update logic
            output_width, output_height = self._compute_output_size_from_polygon(src_points)
            # Update dst_points to match computed size
            dst_points = np.array([
                [0, 0],
                [output_width, 0],
                [output_width, output_height],
                [0, output_height]
            ], dtype=np.float32)
        elif self.output_size is not None:
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
    
    def transform_entire_image_to_marker_plane(self, 
                                               image: np.ndarray,
                                               marker_poses: List[Dict],
                                               plane_extent_meters: Tuple[float, float] = (2.0, 1.0),
                                               output_size: Optional[Tuple[int, int]] = None,
                                               ) -> np.ndarray:
        """
        CURRENTLY DEPRACATED
        
        Project the entire image to a coordinate system defined by the ArUco marker plane.
        This creates a bird's-eye view of the entire scene, not just the marker polygon region.
        
        Args:
            image: Input camera image
            plane_extent_meters: (width, height) of the plane to project in meters (default: 2.0m x 1.0m)
            output_size: Output image size (width, height). If None, computed from plane_extent
            undistort: If True, undistort image before transformation
            
        Returns:
            Warped image showing entire scene from marker plane perspective
        """
        if self.polygon is None or len(self.polygon) != 4 or np.array_equal(self.polygon, [0,0,0,0]):
            return image
        
        # Undistort image if requested
        if self._correct_distortion:
            working_image = cv.undistort(image, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
        else:
            working_image = image
        
        # Use the first marker to define the plane coordinate system
        # The marker's pose defines: origin at marker center, Z-axis normal to marker
        reference_marker = marker_poses[0]
        rvec = reference_marker['rvec']
        tvec = reference_marker['tvec']
        
        # Get rotation matrix
        R, _ = cv.Rodrigues(rvec)
        
        # Define plane coordinate system:
        # - Origin at marker center (tvec)
        # - X-axis: marker's X direction in world
        # - Y-axis: marker's Y direction in world  
        # - Z-axis: marker's Z direction (normal to plane)
        plane_x_axis = R[:, 0]  # Marker's X-axis in camera coordinates
        plane_y_axis = R[:, 1]  # Marker's Y-axis in camera coordinates
        plane_origin = tvec.flatten()
        
        # Determine output size
        if output_size is None:
            # Compute from plane_extent and a reasonable pixel density
            pixels_per_meter = 200  # Adjust based on desired resolution
            output_width = int(plane_extent_meters[0] * pixels_per_meter)
            output_height = int(plane_extent_meters[1] * pixels_per_meter)
        else:
            output_width, output_height = output_size
        
        # Create a grid of 3D points on the marker plane
        # Points are in the marker's local coordinate system
        grid_size = 20  # Number of grid points per dimension
        plane_width, plane_height = plane_extent_meters
        
        # Generate grid points in marker coordinate system (centered at origin)
        x_range = np.linspace(-plane_width/2, plane_width/2, grid_size)
        y_range = np.linspace(-plane_height/2, plane_height/2, grid_size)
        xx, yy = np.meshgrid(x_range, y_range)
        zz = np.zeros_like(xx)  # All points on z=0 plane
        
        # Reshape to list of 3D points
        plane_points_3d = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1).astype(np.float32)
        
        # Transform plane points to camera coordinate system
        # plane_point_camera = R @ plane_point_local + tvec
        plane_points_camera = (R @ plane_points_3d.T).T + plane_origin
        
        # Project 3D points to image coordinates
        plane_points_2d, _ = cv.projectPoints(
            plane_points_3d.reshape(-1, 1, 3),
            rvec, tvec,
            self.new_camera_matrix if self._correct_distortion else self.camera_matrix,
            np.zeros(4) if self._correct_distortion else self.dist_coeffs  # No distortion if already undistorted
        )
        plane_points_2d = plane_points_2d.reshape(-1, 2)
        
        # Create corresponding destination points in output image
        # Map plane coordinates to output image coordinates
        x_norm = (xx.flatten() + plane_width/2) / plane_width  # 0 to 1
        y_norm = (yy.flatten() + plane_height/2) / plane_height  # 0 to 1
        
        dst_points = np.stack([
            x_norm * output_width,
            y_norm * output_height
        ], axis=1).astype(np.float32)
        
        # Filter out points that are outside image bounds
        h, w = working_image.shape[:2]
        valid_mask = (
            (plane_points_2d[:, 0] >= 0) & (plane_points_2d[:, 0] < w) &
            (plane_points_2d[:, 1] >= 0) & (plane_points_2d[:, 1] < h)
        )
        
        if np.sum(valid_mask) < 4:
            # Not enough valid points, return original image
            return image
        
        valid_src = plane_points_2d[valid_mask]
        valid_dst = dst_points[valid_mask]
        
        # Compute homography from image to plane coordinates
        # Using RANSAC for robustness
        H, mask = cv.findHomography(valid_src, valid_dst, 
                                   cv.RANSAC, 
                                   ransacReprojThreshold=5.0)
        
        if H is None:
            return image
        
        # Apply transformation to entire image
        warped_image = cv.warpPerspective(working_image, H, (output_width, output_height),
                                         flags=cv.INTER_LINEAR,
                                         borderMode=cv.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
        
        return warped_image
    
    def transform_image_to_orthographic_plane(self,
                                              image: np.ndarray,
                                              marker_poses: List[Dict],
                                              plane_extent_meters: Tuple[float, float] = (2.0, 1.0),
                                              output_size: Optional[Tuple[int, int]] = None,
                                              pixels_per_meter: float = 200.0) -> np.ndarray:
        """
        Create an improved projection onto the marker plane using 3D pose information.
        
        NOTE: True orthographic projection is not possible with a single perspective camera.
        This method uses a dense grid of 3D points to compute a more accurate homography
        that better accounts for the 3D geometry, reducing (but not eliminating) perspective distortion.
        
        The method:
        1. Creates a dense grid of 3D points on the marker plane
        2. Projects them to image coordinates using the camera's projection matrix
        3. Uses these correspondences to compute a homography
        4. Applies the homography to warp the image
        
        This gives better results than simple 4-point homography but still has some distortion
        because the camera uses perspective projection.
        
        Args:
            image: Input camera image
            marker_poses: List of marker pose dictionaries with 'rvec', 'tvec', 'rotation_matrix'
            plane_extent_meters: (width, height) of the plane in meters
            output_size: Output image size. If None, computed from plane_extent and pixels_per_meter
            pixels_per_meter: Pixel density for output image (default: 200 pixels/meter)
            
        Returns:
            Projected image with reduced perspective distortion
        """
        if len(marker_poses) < 1:
            return image
        
        # Undistort image if requested
        if self._correct_distortion:
            working_image = cv.undistort(image, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
            camera_matrix_to_use = self.new_camera_matrix
            dist_coeffs_to_use = np.zeros(4)  # Already undistorted
        else:
            working_image = image
            camera_matrix_to_use = self.camera_matrix
            dist_coeffs_to_use = self.dist_coeffs
        
        # Use all markers to compute a better plane estimate
        # Average the marker poses to get a more stable plane definition
        if len(marker_poses) >= 4:
            # Use all 4 markers to define the plane more accurately
            # Average the rotation and translation
            rvecs = [pose['rvec'] for pose in marker_poses[:4]]
            tvecs = [pose['tvec'] for pose in marker_poses[:4]]
            
            # Average translation (center of plane)
            avg_tvec = np.mean(tvecs, axis=0)
            
            # For rotation, average the rotation matrices
            R_matrices = []
            for pose in marker_poses[:4]:
                R = pose.get('rotation_matrix')
                if R is None:
                    R, _ = cv.Rodrigues(pose['rvec'])
                R_matrices.append(R)
            
            # Average rotation matrices (simple mean, then re-normalize)
            avg_R = np.mean(R_matrices, axis=0)
            # Re-orthonormalize the rotation matrix
            U, _, Vt = np.linalg.svd(avg_R)
            avg_R = U @ Vt
            
            # Convert back to rotation vector
            avg_rvec, _ = cv.Rodrigues(avg_R)
            
            rvec = avg_rvec
            tvec = avg_tvec
            R = avg_R
        else:
            # Fall back to first marker if not enough markers
            reference_marker = marker_poses[0]
            rvec = reference_marker['rvec']
            tvec = reference_marker['tvec']
            R = reference_marker.get('rotation_matrix')
            if R is None:
                R, _ = cv.Rodrigues(rvec)
        
        # Determine output size
        if output_size is None:
            output_width = int(plane_extent_meters[0] * pixels_per_meter)
            output_height = int(plane_extent_meters[1] * pixels_per_meter)
        else:
            output_width, output_height = output_size
        
        # Create a dense grid of 3D points on the marker plane
        # More points = better homography, but slower computation
        grid_density = 30  # Points per dimension
        plane_width, plane_height = plane_extent_meters
        
        # Generate grid points in marker coordinate system (centered at origin)
        # Output image: j (column) = horizontal, i (row) = vertical
        # Marker: X = right, Y = down
        # Correct mapping: Marker X -> Output j, Marker Y -> Output i
        x_range = np.linspace(-plane_width/2, plane_width/2, grid_density)
        y_range = np.linspace(-plane_height/2, plane_height/2, grid_density)
        xx, yy = np.meshgrid(x_range, y_range)
        zz = np.zeros_like(xx)  # All points on z=0 plane
        
        # Reshape to list of 3D points
        # Marker X -> 3D X, Marker Y -> 3D Y (no swap needed)
        plane_points_3d = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1).astype(np.float32)
        
        # Project 3D points from marker's local coordinate system to image coordinates
        plane_points_2d, _ = cv.projectPoints(
            plane_points_3d.reshape(-1, 1, 3),
            rvec,  # Marker's rotation (transforms from marker coords to camera coords)
            tvec,  # Marker's translation (marker origin in camera coords)
            camera_matrix_to_use,
            dist_coeffs_to_use
        )
        plane_points_2d = plane_points_2d.reshape(-1, 2)
        
        # Create corresponding destination points in output image
        # Map plane coordinates to output image coordinates
        # Marker X (xx) -> Output j (column/horizontal)
        # Marker Y (yy) -> Output i (row/vertical)
        x_norm = (xx.flatten() + plane_width/2) / plane_width   # 0 to 1
        y_norm = (yy.flatten() + plane_height/2) / plane_height  # 0 to 1
        
        # Correct mapping: x_norm -> j, y_norm -> i
        dst_points = np.stack([
            x_norm * output_width,   # j (column) - maps to marker X
            y_norm * output_height   # i (row) - maps to marker Y
        ], axis=1).astype(np.float32)
        
        # Filter out points that are outside image bounds
        h, w = working_image.shape[:2]
        valid_mask = (
            (plane_points_2d[:, 0] >= 0) & (plane_points_2d[:, 0] < w) &
            (plane_points_2d[:, 1] >= 0) & (plane_points_2d[:, 1] < h)
        )
        
        if np.sum(valid_mask) < 4:
            return image
        
        valid_src = plane_points_2d[valid_mask]
        valid_dst = dst_points[valid_mask]
        
        # Compute homography from image to plane coordinates using dense correspondences
        # This should give better results than 4-point homography
        H, mask = cv.findHomography(valid_src, valid_dst, 
                                   cv.RANSAC, 
                                   ransacReprojThreshold=5.0)
        
        if H is None:
            return image
        
        # Apply transformation to entire image
        warped_image = cv.warpPerspective(working_image, H, (output_width, output_height),
                                         flags=cv.INTER_LINEAR,
                                         borderMode=cv.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
        
        return warped_image
    
    def transform_point_to_marker_plane(self,
                                       point_2d: Tuple[float, float],
                                       marker_poses: List[Dict],
                                       marker_size_meters: float,
                                       plane_extent_meters: Tuple[float, float] = (2.0, 1.0),
                                       output_size: Optional[Tuple[int, int]] = None) -> Optional[Tuple[float, float]]:
        """
        Transform a single point from image coordinates to marker plane coordinates.
        
        Args:
            point_2d: (x, y) point in image coordinates
            marker_poses: List of marker pose dictionaries
            marker_size_meters: Physical size of markers in meters
            plane_extent_meters: (width, height) of the plane in meters
            output_size: Output image size. If None, computed from plane_extent
            
        Returns:
            (x, y) in marker plane coordinates, or None if transformation fails
        """
        if len(marker_poses) < 1:
            return None
        
        # This is a simplified version - in practice, you'd want to compute
        # the homography once and reuse it, or use the 3D projection method
        
        # For now, use the same approach as transform_entire_image_to_marker_plane
        # but just for a single point
        reference_marker = marker_poses[0]
        rvec = reference_marker['rvec']
        tvec = reference_marker['tvec']
        
        # Back-project point to 3D ray, then intersect with plane
        # This is more complex - for now, return None as placeholder
        # TODO: Implement proper back-projection and plane intersection
        return None
 
    
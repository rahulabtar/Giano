import numpy as np
import cv2 as cv
from typing import List, Dict, Optional, Tuple
from src.computer_vision.aruco_polygon_detector import ArucoPolygonDetector
from src.core.utils import name_from_midi

class FingerArucoTracker(ArucoPolygonDetector):
    def __init__(self, 
                 camera_matrix:np.ndarray, 
                 dist_coeffs: np.ndarray, 
                 image_size: Optional[Tuple[int, int]] = (640, 480),
                 output_size: Optional[Tuple[int, int]] = (640, 480),
                 use_coordinate_correction: bool = True,
                 correct_camera_distortion: bool = True
                 ):
        """
        Args:
            camera_matrix: The camera projection matrix from calibration
            dist_coeffs: The distortion coefficients from calibration
            image_size: (width, height) of the input image (default: (640, 480))
            output_size: (width, height of the output size for birdseye transform)
            use_coordinate_correction: If True, use coordinate correction for the birdseye transform
            correct_camera_distortion: If True, correct the distortion of the camera
        """
        super().__init__(camera_matrix, dist_coeffs, image_size=image_size, output_size=output_size, correct_distortion=correct_camera_distortion)
        self.fingertip_ids = [4,8,12,16,20]
        self.keyboard_map = None
        self._y_scale_factor = None  # Cache for y-direction scale correction
        self._x_scale_factor = None  # Cache for x-direction scale correction
        self._use_coordinate_correction = use_coordinate_correction  # Flag to enable coordinate correction
        self.horizontal_weight = 2.0  # Weight for horizontal (x) distance (default: 2.0)
        self.vertical_weight = 1.0   # Weight for vertical (y) distance (default: 1.0)

    def set_keyboard_map(self, labeled_keys: List[Dict]):
        """
        Set the keyboard map for the finger aruco tracker.
        """
        for i,key in enumerate(labeled_keys):
            closed_contour = self._ensure_contour_closed(key['contour'])
            if self._is_contour_closed(closed_contour):
                labeled_keys[i]['contour'] = closed_contour
            else:
                print("Contour is not closed")
                return False
        self.keyboard_map = labeled_keys
        # Recompute scale factors when keyboard map changes
        self._x_scale_factor = None
        self._y_scale_factor = None

    
    def get_finger_keys(self, hand_landmarks: List, 
                       ) -> Dict[int, Optional[int]]:
        """
        Get piano keys for all fingertip positions.
        
        Args:
            hand_landmarks: List of hand landmark data from MediaPipe
            
        Returns:
            Dictionary mapping finger_id to key information (or None)
        """
        finger_keys = {}
        
        # Get normalized positions for all fingertips
        for landmark in hand_landmarks:
            lm_id, x_px, y_px = landmark[0], landmark[1], landmark[2]
            
            if lm_id in self.fingertip_ids:
                if x_px != 0 and y_px != 0:

                    midi_note = self.find_closest_key(x_px, y_px)
                    if midi_note is not None:
                        finger_keys[lm_id] = midi_note

        return finger_keys
    
    def find_closest_key(self, x_px: float, y_px: float, method: str = 'default') -> Optional[int]:
        """
        Find the closest key to a given (x_px, y_px) point in image space.
        
        Args:
            x_px: X coordinate in image space (pixels)
            y_px: Y coordinate in image space (pixels)
            method: Method to use for finding closest key. Options:
                - 'default' or 'centroid': Uses weighted distance to centroid (default)
                - 'boundary': Uses distance to polygon boundary
                - 'y_correction': Uses y-direction corrected distance metric
                - 'full_correction': Uses full coordinate correction for both x and y
            
        Returns:
            MIDI note number of the closest key, 
            or None if no key is found
        """
        if self.keyboard_map is None:
            return None
        
        # Method: 'default' or 'centroid'
        if method == 'default' or method == 'centroid':
            # Transform point from image space to birdseye space
            aruco_coords = self.transform_point_from_image_to_birdseye((x_px, y_px))
            point = np.array([aruco_coords[0], aruco_coords[1]])
            
            closest_midi_note = None
            min_distance = float('inf')

            for key in self.keyboard_map:
                result = cv.pointPolygonTest(key['contour'], point, measureDist=False)
                if result > 0:
                    return key['midi_note']
                
                # Use weighted distance: weight horizontal more than vertical
                centroid = np.array(key['centroid'])
                dx = point[0] - centroid[0]
                dy = point[1] - centroid[1]
                distance = self._weighted_distance(dx, dy)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_midi_note = key['midi_note']
            
            return closest_midi_note
        
        # Method: 'boundary'
        elif method == 'boundary':
            # Transform point from image space to birdseye space
            aruco_coords = self.transform_point_from_image_to_birdseye((x_px, y_px))
            point = np.array([aruco_coords[0], aruco_coords[1]], dtype=np.float32)
            
            closest_midi_note = None
            min_distance = float('inf')

            for key in self.keyboard_map:
                # Use pointPolygonTest with measureDist=True to get signed distance
                # Positive = inside polygon, negative = outside, value = distance to boundary
                distance = cv.pointPolygonTest(key['contour'], point, measureDist=True)
                
                if distance > 0:
                    # Point is inside the key - return immediately
                    return key['midi_note']
                
                # For points outside, use absolute distance (negative value)
                # This gives distance to the polygon boundary, which accounts for projection distortion
                abs_distance = abs(distance)
                if abs_distance < min_distance:
                    min_distance = abs_distance
                    closest_midi_note = key['midi_note']
            
            return closest_midi_note
        
        # Method: 'y_correction'
        elif method == 'y_correction':
            # Transform point from image space to birdseye space
            aruco_coords = self.transform_point_from_image_to_birdseye((x_px, y_px))
            point = np.array([aruco_coords[0], aruco_coords[1]], dtype=np.float32)
            
            # Get y-scale correction factor
            y_scale = self._get_y_scale_factor()
            
            closest_midi_note = None
            min_distance = float('inf')

            for key in self.keyboard_map:
                # Check if point is inside key
                result = cv.pointPolygonTest(key['contour'], point, measureDist=False)
                if result > 0:
                    return key['midi_note']
                
                # Compute distance with y-correction and horizontal weighting
                # Apply scale factor to y-component to account for perspective distortion
                centroid = np.array(key['centroid'])
                dx = point[0] - centroid[0]
                dy = (point[1] - centroid[1]) * y_scale  # Apply correction to y-component
                
                # Use weighted distance (horizontal weighted more than vertical)
                distance = self._weighted_distance(dx, dy)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_midi_note = key['midi_note']
            
            return closest_midi_note
        
        # Method: 'full_correction'
        elif method == 'full_correction':
            # Transform with coordinate correction
            corrected_x, corrected_y = self.transform_point_with_correction(x_px, y_px)
            point = np.array([corrected_x, corrected_y], dtype=np.float32)
            
            closest_midi_note = None
            min_distance = float('inf')

            for key in self.keyboard_map:
                # Apply same correction to key centroid for fair comparison
                # Note: This assumes keys were detected in uncorrected space
                # For best results, keys should be stored with corrected coordinates
                centroid = np.array(key['centroid'])
                if self._use_coordinate_correction:
                    x_scale, y_scale = self._get_scale_factors()
                    corrected_centroid = np.array([centroid[0] * x_scale, centroid[1] * y_scale])
                else:
                    corrected_centroid = centroid
                
                # Check if point is inside key (using original contour - this is approximate)
                result = cv.pointPolygonTest(key['contour'], point, measureDist=False)
                if result > 0:
                    return key['midi_note']
                
                # Compute distance with corrected coordinates and horizontal weighting
                dx = point[0] - corrected_centroid[0]
                dy = point[1] - corrected_centroid[1]
                distance = self._weighted_distance(dx, dy)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_midi_note = key['midi_note']
            
            return closest_midi_note
        
        else:
            raise ValueError(f"Unknown method: {method}. Must be one of: 'default', 'centroid', 'boundary', 'y_correction', 'full_correction'")
    
    def _compute_scale_factors(self) -> Tuple[float, float]:
        """
        Compute the x and y-direction scale correction factors from the perspective transformation.
        This accounts for non-uniform scaling in the bird's-eye view transformation.
        
        Returns:
            Tuple of (x_scale_factor, y_scale_factor) to normalize the coordinate system
        """
        if self.birdseye_perspective_matrix is None:
            return 1.0, 1.0
        
        # Compute how unit vectors in x and y directions transform
        # This tells us the scale factors in each direction
        M = self.birdseye_perspective_matrix
        
        # Transform unit vectors in homogeneous coordinates
        # Unit vector in x-direction: [1, 0, 1]
        x_unit = np.array([1, 0, 1], dtype=np.float32)
        # Unit vector in y-direction: [0, 1, 1]
        y_unit = np.array([0, 1, 1], dtype=np.float32)
        
        # Transform and convert back from homogeneous coordinates
        x_transformed = M @ x_unit
        x_transformed = x_transformed[:2] / x_transformed[2]
        
        y_transformed = M @ y_unit
        y_transformed = y_transformed[:2] / y_transformed[2]
        
        # Compute the magnitude of the transformed vectors
        x_scale = np.linalg.norm(x_transformed)
        y_scale = np.linalg.norm(y_transformed)
        
        # Normalize to the average scale to preserve aspect ratio
        avg_scale = (x_scale + y_scale) / 2.0
        
        if x_scale > 0 and y_scale > 0:
            x_factor = avg_scale / x_scale
            y_factor = avg_scale / y_scale
            return x_factor, y_factor
        
        return 1.0, 1.0
    
    def _get_scale_factors(self) -> Tuple[float, float]:
        """Get cached scale factors, computing them if necessary."""
        if self._x_scale_factor is None or self._y_scale_factor is None:
            self._x_scale_factor, self._y_scale_factor = self._compute_scale_factors()
        return self._x_scale_factor, self._y_scale_factor
    
    def _get_y_scale_factor(self) -> float:
        """Get cached y-scale factor, computing it if necessary."""
        _, y_factor = self._get_scale_factors()
        return y_factor
    
    def _weighted_distance(self, dx: float, dy: float) -> float:
        """
        Compute weighted L2 distance, giving more weight to horizontal (x) distance.
        This accounts for y-direction distortion in perspective projection.
        
        Args:
            dx: Horizontal distance (x-direction)
            dy: Vertical distance (y-direction)
            
        Returns:
            Weighted distance: sqrt((w_x * dx)^2 + (w_y * dy)^2)
        """
        return np.sqrt((self.horizontal_weight * dx)**2 + (self.vertical_weight * dy)**2)
    

    
    def transform_point_with_correction(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """
        Transform a point from image space to birdseye space with coordinate correction.
        This applies scale factors to account for perspective distortion in both x and y.
        
        Args:
            x_px: X coordinate in image space (pixels)
            y_px: Y coordinate in image space (pixels)
            
        Returns:
            Tuple of (corrected_x, corrected_y) in birdseye space
        """
        # Transform point from image space to birdseye space
        aruco_coords = self.transform_point_from_image_to_birdseye((x_px, y_px))
        
        if self._use_coordinate_correction:
            # Get scale factors and apply correction
            x_scale, y_scale = self._get_scale_factors()
            corrected_x = aruco_coords[0] * x_scale
            corrected_y = aruco_coords[1] * y_scale
            return corrected_x, corrected_y
        else:
            return aruco_coords[0], aruco_coords[1]
    
 

    
    def measure_distance_to_key(self, x_px: float, y_px: float, midi_note: int) -> Optional[float]:
        """
        Measure the weighted distance to a given key in image space.
        Horizontal distance is weighted more than vertical distance.
        """
        if self.keyboard_map is None:
            return None
        
        for key in self.keyboard_map:
            if key['midi_note'] == midi_note:
                aruco_x, aruco_y = self.transform_point_from_image_to_birdseye((x_px, y_px))
                point = np.array([aruco_x, aruco_y])
                centroid = np.array(key['centroid'])
                dx = point[0] - centroid[0]
                dy = point[1] - centroid[1]
                return self._weighted_distance(dx, dy)
    
    
    def _is_contour_closed(self, contour: np.ndarray) -> bool:
        """
        Check if a contour is closed (first and last points are the same).
        
        Args:
            contour: Contour array with shape (N, 1, 2)
            
        Returns:
            True if contour is closed, False otherwise
        """
        if len(contour) < 3:
            return False
        first_point = contour[0, 0]
        last_point = contour[-1, 0]
        return np.allclose(first_point, last_point, atol=1e-6)
    
    def _ensure_contour_closed(self, contour: np.ndarray) -> np.ndarray:
        """
        Ensure a contour is closed by adding the first point at the end if needed.
        
        Args:
            contour: Contour array with shape (N, 1, 2)
            
        Returns:
            Closed contour array
        """
        if self._is_contour_closed(contour):
            return contour
        
        # Add first point at the end to close the contour
        first_point = contour[0:1, :, :].copy()
        closed_contour = np.vstack([contour, first_point])
        return closed_contour
    


    def _map_coords_to_uncropped(self, coords_or_contour, y_offset: int):
        """
        Map coordinates or contours from cropped image space back to uncropped image space.
        
        Args:
            coords_or_contour: Can be:
                - Tuple (x, y) for a single point
                - NumPy array with shape (N, 1, 2) for contour/polygon
                - Tuple (x, y, w, h) for bbox
            y_offset: Y offset to add (pixels cropped from top)
            
        Returns:
            Same type as input but with y coordinates adjusted
        """
        if isinstance(coords_or_contour, tuple):
            if len(coords_or_contour) == 2:
                # Single point (x, y)
                x, y = coords_or_contour
                return (x, y + y_offset)
            elif len(coords_or_contour) == 4:
                # Bbox (x, y, w, h)
                x, y, w, h = coords_or_contour
                return (x, y + y_offset, w, h)
        elif isinstance(coords_or_contour, np.ndarray):
            # Contour or polygon array (N, 1, 2)
            mapped = coords_or_contour.copy()
            mapped[:, 0, 1] += y_offset  # Add offset to all y coordinates
            return mapped
        
        return coords_or_contour


    def draw_finger_positions(self, image, finger_positions, hand_landmarks:List):
        """Draw finger positions and coordinates on the image."""
        for landmark in hand_landmarks:
            lm_id, x_px, y_px = landmark[0], landmark[1], landmark[2]
            
            if lm_id in finger_positions:
                u, v = finger_positions[lm_id]
                
                # Draw finger position
                cv.circle(image, (x_px, y_px), 8, (255, 0, 0), -1)
                
                # Draw coordinates
                coord_text = f"({u:.2f}, {v:.2f})"
                cv.putText(image, coord_text, (x_px + 10, y_px - 10),
                         cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return image


    def draw_birdseye_keys(self, image, landmarks:List, finger_keys:List):
        """
        Draw birdseye keys on the image.
        Args:
            image: The image to draw on.
            landmarks: Input from MediaPipe. In image space.
            finger_keys: Dictionary mapping finger_id to key information (or None)
        Returns:
            The image with the birdseye keys drawn on it.
        """
        birdseye_image = self.transform_image_to_birdseye(image, use_polygon_aspect_ratio=True)
        for key in self.keyboard_map:
            birdseye_image = cv.polylines(birdseye_image, [key['contour']], isClosed=True, color=(0, 255, 0), thickness=2)
            birdseye_image = cv.putText(birdseye_image, key['name'], key['centroid'], cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
        for landmark in landmarks:
            # because the landmark is in image space, we need to transform it to birdseye space
            # aruco_x, aruco_y = self.transform_point_from_image_to_birdseye((landmark[1], landmark[2]))
            aruco_x, aruco_y = self.transform_point_with_correction(landmark[1], landmark[2])
            if landmark[0] in finger_keys.keys():
                midi_note = finger_keys[landmark[0]]
                key = next((k for k in self.keyboard_map if k['midi_note'] == midi_note), None)
                cv.circle(birdseye_image, (int(round(aruco_x)), int(round(aruco_y))), 8, (255, 0, 0), -1)
                cv.line(birdseye_image, (int(round(aruco_x)), int(round(aruco_y))), (key['centroid'][0], key['centroid'][1]), (0, 255, 0), 2)
        return birdseye_image
    


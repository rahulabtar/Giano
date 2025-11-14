"""
Piano Calibration Class

This module provides a PianoCalibration class for detecting and labeling piano keys
using ridge detection and ArUco marker-based perspective correction.
"""

import cv2 as cv
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.computer_vision.aruco_polygon_detector import ArucoPolygonDetector
from src.computer_vision.aruco_pose_tracker import ArucoPoseTracker, TrackingMode
from src.core.constants import CAMERA_CALIBRATION_PATH, MARKER_IDS, MARKER_SIZE, IN_TO_METERS

class PianoCalibration:
    """Class for calibrating and detecting piano keys using ridge detection."""
    
    def __init__(self, 
                 camera_matrix: np.ndarray,
                 dist_coeffs: np.ndarray,
                 pose_tracker: ArucoPoseTracker,
                 polygon_detector: ArucoPolygonDetector,
                 marker_ids: List[int],
                 marker_size_meters: float):
      """
      Initialize the PianoCalibration class.
      
      Args:
        camera_matrix: Camera calibration matrix
        dist_coeffs: Distortion coefficients
        pose_tracker: ArucoPoseTracker instance for marker detection
        polygon_detector: ArucoPolygonDetector instance for perspective transform
        marker_ids: List of ArUco marker IDs to detect
        marker_size_meters: Size of ArUco markers in meters
      """
      self.camera_matrix = camera_matrix
      self.dist_coeffs = dist_coeffs
      self.pose_tracker = pose_tracker
      self.polygon_detector = polygon_detector
      self.marker_ids = marker_ids
      self.marker_size_meters = marker_size_meters
    
    def _calibration_loop(self, cap:cv.VideoCapture) -> tuple[int, np.ndarray]:
      """
      Calibration loop that allows capturing and confirming images.
  
      Args:
        cap: VideoCapture object
      """
      captured_image = None
      
      while True:
        success, image = cap.read()
        if not success: 
          print("Error reading from camera")
          return (2, None)
        
        # If we have a captured image, show it with confirmation options
        if captured_image is not None:
          cv.imshow("Press c to confirm or p to retake", captured_image)
          key = cv.waitKey(1) & 0xFF
          
          if key == ord('c'):
            cv.destroyAllWindows()
            self.image = captured_image
            return (1, captured_image)
          elif key == ord('p'):
            captured_image = None  # Go back to live view
            cv.destroyWindow("Press c to confirm or p to retake")
          elif key == ord('q'):
            cv.destroyAllWindows()
            return (2, None)
        
        # Live view mode - show current camera feed
        else:
          cv.imshow("Press p to capture or q to quit", image)
          key = cv.waitKey(1) & 0xFF
          
          if key == ord('p'):
            captured_image = image.copy()  # Capture current frame
            cv.destroyWindow("Press p to capture or q to quit")
          elif key == ord('q'):
            cv.destroyAllWindows()
            return (2, None)

    def detect_ridge_mask(self, image: np.ndarray, 
                         median_blur_size: int = 3,
                         ridge_pad_size: int = 5,
                         debug_mode: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect piano key boundaries using ridge detection.
        
        Args:
            image: Input BGR image from camera
            median_blur_size: Size of median blur kernel (default: 3)
            ridge_pad_size: Padding size for ridge detection to avoid edge artifacts (default: 5)
            debug_mode: If True, display intermediate results (default: False)
            
        Returns:
            Tuple of (grayscale_image, ridge_mask) where:
                - grayscale_image: Grayscale bird's-eye view of the piano
                - ridge_mask: Binary mask with key boundaries (255) and keys (0)
        """
        # Get ArUco marker poses and polygon
        poses = self.pose_tracker.get_marker_poses(image, self.marker_size_meters)
        success, marker_list_2d = self.polygon_detector.get_marker_polygon(
            self.marker_ids, poses, store_polygon=True
        )
        
        # Check if markers were found
        if not success or np.array_equal(marker_list_2d, [0, 0, 0, 0]):
            raise ValueError("ArUco markers not found in image!")
        
        print("ArUco markers found! Processing piano image...")
        start_time = time.time()
        
        # Transform to bird's eye view
        warped = self.polygon_detector.transform_image_to_birdseye(image, undistort=True)
        transform_time = time.time() - start_time
        print(f"Image transformation (birdseye): {transform_time:.4f} seconds")
        
        # Convert to grayscale
        gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        
        # Apply median blur to reduce noise
        # Pad first to avoid edge artifacts
        pad_size = 2
        gray_padded = cv.copyMakeBorder(gray, pad_size, pad_size, pad_size, pad_size, 
                                       cv.BORDER_REPLICATE)
        med_image_padded = cv.medianBlur(gray_padded, median_blur_size)
        med_image = med_image_padded[pad_size:-pad_size, pad_size:-pad_size]
        
        # Ridge detection
        start_time = time.time()
        
        # Pad image to avoid edge artifacts in ridge detection
        med_image_padded = cv.copyMakeBorder(med_image, ridge_pad_size, ridge_pad_size, 
                                            ridge_pad_size, ridge_pad_size, 
                                            cv.BORDER_REPLICATE)
        
        ridge = cv.ximgproc.RidgeDetectionFilter_create()
        ridge_img_padded = ridge.getRidgeFilteredImage(med_image_padded)
        
        # Remove padding to get back to original size
        ridge_img = ridge_img_padded[ridge_pad_size:-ridge_pad_size, 
                                    ridge_pad_size:-ridge_pad_size]
        
        
        # Threshold with Otsu to create binary mask
        _, ridge_bin = cv.threshold(ridge_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        if debug_mode:
            cv.imshow('ridge otsu result', ridge_bin)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
        return gray, ridge_bin
    
    def _crop_marker_regions(self, mask: np.ndarray, 
                           mask_margin_pct: float = 0.1) -> Tuple[np.ndarray, int]:
        """
        Crop the mask to remove ArUco marker regions from top and bottom.
        
        Args:
            mask: Input mask image
            mask_margin_pct: Percentage of image height to crop from top/bottom (default: 0.1)
            
        Returns:
            Tuple of (cropped_mask, y_offset) where y_offset is pixels cropped from top
        """
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        h, w = mask.shape
        
        # Calculate mask size for corners (only cropping top/bottom)
        mask_y = int(h * mask_margin_pct)
        
        cropped = mask[mask_y:h-mask_y, :].copy()
        
        return cropped, mask_y
    
    def _label_keys_from_boundary_mask(boundary_mask: np.ndarray,
                                  start_midi: int,
                                  expected_keys: int | None = None,
                                  close_px: int = 3,
                                  debug_mode: bool = False) -> List[Dict]:
      """
      boundary_mask: uint8 image with aruco markers cropped out, 255 at key boundaries, 0 elsewhere
      start_midi: MIDI note of the left-most key in view (e.g., 21 for A0, 48 for C3)
      expected_keys: optional sanity check on count
      close_px: thickness to ensure separators are closed

      returns list of dicts: [{name, midi, is_black, bbox, contour, centroid}, ...]
      """
      output_dir = os.path.join(os.path.dirname(__file__), "calibration_output", f"final_masks")

      h, w = boundary_mask.shape

      # Remove small noise using connected components instead of opening
      # Opening breaks connections, so we filter by area instead
      # This preserves line segments while removing isolated noise pixels
      num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(boundary_mask, connectivity=8)
      
      # Filter out small noise regions (keep only regions above threshold)
      # Adjust min_area based on your noise level - smaller values keep more, larger removes more
      min_area = 100  # Minimum pixels to keep (removes isolated 1-2 pixel noise)
      walls_cleaned = np.zeros_like(boundary_mask)
      
      for label_id in range(1, num_labels):  # Skip background (label 0)
        area = stats[label_id, cv.CC_STAT_AREA]
        if area >= min_area:
          # Keep this region
          walls_cleaned[labels == label_id] = 255


      # Now proceed with closing operations to connect fragments
      # Skip opening - it breaks connections. Use closing to connect instead.
      walls_horizontal = walls_cleaned.copy()

      # 1) Vertical closing: connects broken vertical boundary segments
      # Use a tall vertical line kernel to connect vertical gaps (most important for piano keys)
      vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 5))  # 1 wide, 10 tall
      walls_vertical_closed = cv.morphologyEx(walls_horizontal, cv.MORPH_CLOSE, vertical_kernel)
      
      # Apply multiple iterations for better connection of stubborn gaps
      walls_vertical_closed = cv.morphologyEx(walls_vertical_closed, cv.MORPH_CLOSE, vertical_kernel)
      
      if debug_mode:
        cv.imshow('walls_vertical_closed', walls_vertical_closed)
        cv.waitKey(0)
        cv.destroyAllWindows()

      # 2) Horizontal closing: connects broken horizontal boundary segments
      # Use a WIDE horizontal line kernel to connect horizontal gaps (top/bottom edges)
      # CRITICAL: Increased from 3 to 60 - the previous kernel was way too small!
      horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))  # 60 wide, 1 tall
      walls_connected = cv.morphologyEx(walls_vertical_closed, cv.MORPH_CLOSE, horizontal_kernel)
      
      # Apply multiple iterations for better connection
      walls_connected = cv.morphologyEx(walls_connected, cv.MORPH_CLOSE, horizontal_kernel)
      
      if debug_mode:
        cv.imshow('walls_vertical_and_horizontal', walls_connected)
        cv.waitKey(0)
        cv.destroyAllWindows()  
      

      # 3) Invert to get key regions as 1s
      regions = cv.bitwise_not(walls_connected)
      regions = (regions > 0).astype(np.uint8)
      
      # 4) Opening on key regions: removes small artifacts inside keys (white specks)
      # This cleans up noise that appears as small white dots in the black key regions
      artifact_removal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
      regions = cv.morphologyEx(regions, cv.MORPH_OPEN, artifact_removal_kernel, iterations=1)
      

      # 5) Slight dilation to recover area lost during wall dilation
      # This makes contours trace closer to actual key boundaries
      if close_px > 1:
        dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (close_px-1, close_px-1))
        regions = cv.dilate(regions, dilate_kernel, iterations=1)



      # 3) Connected components
      # num, labels, stats, cents = cv.connectedComponentsWithStats(regions, connectivity=4, ltype=cv.CV_32S)
      # regions is the filled key regions (0/1 or 0/255)
      contours, _ = cv.findContours((regions*255).astype(np.uint8),
                                    cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
      
      countour_image = cv.cvtColor(boundary_mask.copy(), cv.COLOR_GRAY2BGR)
      contour_image = cv.drawContours(countour_image, contours, -1, (255, 0, 0), 2)
      cv.imshow('contour image', contour_image)
      cv.waitKey(0)
      cv.destroyAllWindows()

      keys = []
      NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

      def name_from_midi(m: int) -> Tuple[str,bool]:
        n = NAMES[m % 12]
        octave = (m // 12) - 1
        return f"{n}{octave}", ('#' in n)

      def is_square_like(cnt, bbox):
          """Check if contour has wide aspect ratio."""
          x, y, w, h = bbox
          aspect = w / max(h, 1)
          # Square-like: aspect ratio between 0.7 and 1.4
          if aspect > 0.5:
            return True
          return False


      for cnt in contours:
          area = cv.contourArea(cnt)
          x, y, w, h = cv.boundingRect(cnt)

          # Filter out small artifacts and overly large regions
          # Small artifacts: less than 0.1% of image area (noise, specks)
          # Large regions: more than 40% of image area (likely errors)
          min_area = 0.01 * boundary_mask.size  # Minimum area threshold
          max_area = 0.4 * boundary_mask.size    # Maximum area threshold
          
          if area < min_area or area > max_area:
              continue
          
          # Also filter by aspect ratio to catch elongated artifacts
          aspect = w / h if h > 0 else 0
          if aspect > 0.5:
              continue

          # The spatial moments
          # are computed as:
          # m00 = sum(region)
          # m01 = sum(region * y)
          # m10 = sum(region * x)
          # m11 = sum(region * x * y)

          M = cv.moments(cnt)
          if M['m00'] == 0:
              continue
          cx = int(M['m10'] / M['m00'])
          cy = int(M['m01'] / M['m00'])
          
          # Filter out square-like shapes (ArUco markers that might have escaped cropping)
          # if is_square_like(cnt, (x, y, w, h)):
          #     continue
          
          # Polygon approximation - smaller epsilon = closer fit to contour
          # epsilon controls max distance between original curve and approximation
          # 0.001 = 0.1% of perimeter (very close), 0.005 = 0.5% (close), 0.01 = 1% (moderate)
          poly = cv.approxPolyDP(cnt, epsilon=0.005 * cv.arcLength(cnt, True), closed=True)

          keys.append({
              'contour': cnt,        # full contour
              'poly': poly,          # simplified polygon
              'centroid': (cx, cy),
              'bbox': (x, y, w, h),
              'height': h,
              'y': y,
              'area': area
          }) 
      # Sort keys by x (horizontal start of boundingRect)
      keys.sort(key=lambda k: k['bbox'][0])
      for k in keys:
        midi = start_midi + keys.index(k)
        name, is_black_from_midi = name_from_midi(midi)
        k['name'] = name
        k['midi_note'] = midi
        k['is_black'] = is_black_from_midi
      
      if not keys:
        print("No keys found")
        return []
      
      if expected_keys is not None and not np.isclose(len(keys), expected_keys, atol=2):
        print(f"Expected {expected_keys} keys but found {len(keys)}")
        return []
      return keys
   
    def _label_keys_from_boundary_mask(self, 
                                     boundary_mask: np.ndarray,
                                     start_midi: int,
                                     expected_keys: Optional[int] = None,
                                     close_px: int = 3,
                                     debug_mode: bool = False) -> List[Dict]:
        """
        Extract and label individual piano keys from a boundary mask.
        
        Args:
            boundary_mask: uint8 image with 255 at key boundaries, 0 elsewhere
            start_midi: MIDI note of the left-most key in view (e.g., 21 for A0, 48 for C3)
            expected_keys: Optional sanity check on key count
            close_px: Thickness to ensure separators are closed (default: 3)
            debug_mode: If True, display intermediate results (default: False)
        Returns:
            List of dicts: [{name, midi_note, is_black, bbox, contour, poly, centroid, height, y, area}, ...]
        """
        h, w = boundary_mask.shape
        
        # 1) Thicken boundaries to close small gaps
        k = cv.getStructuringElement(cv.MORPH_RECT, (close_px, close_px))
        walls = cv.dilate(boundary_mask, k, iterations=1)
        
        if debug_mode:
            cv.imshow('closed walls', walls)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
        # 2) Invert to get key regions as 1s
        regions = cv.bitwise_not(walls)
        regions = (regions > 0).astype(np.uint8)
        
        # 3) Dilate regions slightly to recover area lost during wall dilation
        if close_px > 1:
            dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (close_px-1, close_px-1))
            regions = cv.dilate(regions, dilate_kernel, iterations=1)
        
        # 4) Find contours of key regions
        contours, _ = cv.findContours((regions * 255).astype(np.uint8),
                                     cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        keys = []
        NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        def name_from_midi(m: int) -> Tuple[str, bool]:
            n = NAMES[m % 12]
            octave = (m // 12) - 1
            return f"{n}{octave}", ('#' in n)
        
        for cnt in contours:
            area = cv.contourArea(cnt)
            x, y, w, h = cv.boundingRect(cnt)
            
            M = cv.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Optional simplification (keeps notches)
            poly = cv.approxPolyDP(cnt, epsilon=0.01 * cv.arcLength(cnt, True), closed=True)
            
            keys.append({
                'contour': cnt,        # full contour
                'poly': poly,          # simplified polygon
                'centroid': (cx, cy),
                'bbox': (x, y, w, h),
                'height': h,
                'y': y,
                'area': area
            })
        
        # Sort keys by x (horizontal start of boundingRect)
        keys.sort(key=lambda k: k['bbox'][0])
        
        # Assign MIDI names
        for idx, k in enumerate(keys):
            midi = start_midi + idx
            name, is_black_from_midi = name_from_midi(midi)
            k['name'] = name
            k['midi_note'] = midi
            k['is_black'] = is_black_from_midi
        
        if expected_keys is not None and len(keys) != expected_keys:
            print(f"Warning: Expected {expected_keys} keys but found {len(keys)}")
        
        return keys
    
    def map_coords_to_original(self, coords_or_contour, y_offset: int):
        """
        Map coordinates or contours from cropped image space back to original image space.
        
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
    
    def is_contour_closed(self, contour: np.ndarray) -> bool:
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
    
    def ensure_contour_closed(self, contour: np.ndarray) -> np.ndarray:
        """
        Ensure a contour is closed by adding the first point at the end if needed.
        
        Args:
            contour: Contour array with shape (N, 1, 2)
            
        Returns:
            Closed contour array
        """
        if self.is_contour_closed(contour):
            return contour
        
        # Add first point at the end to close the contour
        first_point = contour[0:1, :, :].copy()
        closed_contour = np.vstack([contour, first_point])
        return closed_contour
    
    def is_point_inside_contour(self, point: Tuple[int, int], contour: np.ndarray) -> bool:
      """
      Check if a point is inside a contour using cv.pointPolygonTest.
      
      Args:
          point: (x, y) tuple of the point to test
          contour: Contour array with shape (N, 1, 2)
          
      Returns:
          True if point is inside the contour, False otherwise
      """
      # Ensure contour is closed for accurate testing
      closed_contour = self.ensure_contour_closed(contour)
      
      # pointPolygonTest returns:
      # > 0 if point is inside
      # = 0 if point is on the boundary
      # < 0 if point is outside
      result = cv.pointPolygonTest(closed_contour, point, measureDist=False)
      return result >= 0
    
    def find_key_containing_point(self, point: Tuple[int, int], 
                                  labeled_keys: List[Dict],
                                  y_offset: int = 0) -> Optional[Dict]:
      """
      Find which key (if any) contains the given point.
      
      Args:
          point: (x, y) tuple in original image coordinates
          labeled_keys: List of key dictionaries from label_keys_from_boundary_mask
          y_offset: Y offset if keys are from cropped image (default: 0)
          
      Returns:
          Key dictionary if point is inside a key, None otherwise
      """
      # Adjust point for cropped image space if needed
      if y_offset > 0:
          adjusted_point = (point[0], point[1] - y_offset)
      else:
          adjusted_point = point
      
      for key in labeled_keys:
          contour = key['contour']
          if self.is_point_inside_contour(adjusted_point, contour):
              return key
      
      return None
    
    def draw_labeled_keys(self, 
                         image: np.ndarray, 
                         labeled: List[Dict], 
                         y_offset: int = 0) -> np.ndarray:
        """
        Draw labeled keys on the image.
        
        Args:
            image: Original (uncropped) image to draw on (grayscale or BGR)
            labeled: List of key dictionaries from label_keys_from_boundary_mask
            y_offset: Y offset to add if keys are from cropped image (default: 0)
            
        Returns:
            Image with drawn keys (BGR)
        """
        if len(image.shape) == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        
        for p_key in labeled:
            # Map bbox coordinates back to original image space
            x, y, w, h = self.map_coords_to_original(p_key['bbox'], y_offset)
            pt1 = (int(x), int(y))
            pt2 = (int(x+w), int(y+h))
            
            # Map centroid back to original image space
            cx, cy = self.map_coords_to_original(p_key['centroid'], y_offset)
            
            # Map polygon and contour back to original image space
            poly_original = self.map_coords_to_original(p_key['poly'], y_offset)
            contour_original = self.map_coords_to_original(p_key['contour'], y_offset)
            
            # Draw on original image
            image = cv.circle(image, (cx, cy), 5, (255, 0, 127), -1)  # Magenta centroid
            image = cv.putText(image, p_key['name'], (cx, cy-10), 
                             cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            image = cv.drawContours(image, [contour_original], -1, (0, 255, 0), 2)  # Green contour
        
        return image
    
    def get_piano_calibration(self, 
            cap: cv.VideoCapture,
            start_midi: int = 48,
            mask_margin_pct: float = 0.09,
            debug_mode: bool = False) -> Dict:
      """
      Complete pipeline: detect ridge mask, crop markers, label keys, and draw results.
      
      Args:
          start_midi: MIDI note of the left-most key (default: 48 for C3)
          mask_margin_pct: Percentage to crop from top/bottom for marker removal (default: 0.09)
          debug_mode: If True, display intermediate results (default: False)
          
      Returns:
          Dictionary with keys:
              - 'gray': Grayscale bird's-eye view
              - 'ridge_mask': Binary ridge mask (before cropping)
              - 'cropped_mask': Ridge mask after cropping
              - 'y_offset': Y offset from cropping
              - 'labeled_keys': List of labeled key dictionaries
              - 'annotated_image': Image with keys drawn
      """
      result, image = self._calibration_loop(cap)
      if result == 2:
        return None
      # Detect ridge mask
      gray, ridge_mask = self.detect_ridge_mask(image, debug_mode=debug_mode)
      
      # Crop marker regions
      cropped_mask, y_offset = self._crop_marker_regions(ridge_mask, mask_margin_pct)
      
      # Label keys
      labeled_keys = self._label_keys_from_boundary_mask(
          cropped_mask, start_midi, close_px=3
      )
      
      # Draw labeled keys on grayscale image
      annotated_image = self.draw_labeled_keys(gray, labeled_keys, y_offset)
      
      return {
        'gray': gray,
        'ridge_mask': ridge_mask,
        'cropped_mask': cropped_mask,
        'y_offset': y_offset,
        'labeled_keys': labeled_keys,
        'annotated_image': annotated_image
      }


def main():
  calib_npz = np.load(CAMERA_CALIBRATION_PATH)
  camera_matrix = calib_npz["camera_matrix"]
  dist_coeffs = calib_npz["dist_coeffs"]
  pose_tracker = ArucoPoseTracker(camera_matrix, dist_coeffs, mode=TrackingMode.STATIC, marker_ids=MARKER_IDS)
  polygon_detector = ArucoPolygonDetector(camera_matrix, dist_coeffs)
  marker_size_meters = MARKER_SIZE*IN_TO_METERS
  piano_calibrator = PianoCalibration(camera_matrix, dist_coeffs, pose_tracker, polygon_detector, MARKER_IDS, marker_size_meters)
  
  cap = cv.VideoCapture(0)
  calibration_result = piano_calibrator.get_piano_calibration(cap)

  cv.imshow('annotated_image', calibration_result['annotated_image'])
  cv.waitKey(0)
  cv.destroyAllWindows()


if __name__ == "__main__":
    main()


"""
Piano Calibration Class

This module provides a PianoCalibration class for detecting and labeling piano keys
using ridge detection and ArUco marker-based perspective correction.
"""

import cv2 as cv
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.computer_vision.finger_aruco_tracker import FingerArucoTracker
from src.computer_vision.aruco_pose_tracker import ArucoPoseTracker, TrackingMode
from src.core.constants import CAMERA_CALIBRATION_PATH, MARKER_IDS, MARKER_SIZE, IN_TO_METERS
from src.core.utils import name_from_midi

class PianoCalibration:
  """Class for calibrating and detecting piano keys using ridge detection."""
  
  def __init__(self, 
                camera_matrix: np.ndarray,
                dist_coeffs: np.ndarray,
                pose_tracker: ArucoPoseTracker,
                finger_tracker: FingerArucoTracker,
                marker_ids: List[int],
                marker_size_meters: float):
    """
    Initialize the PianoCalibration class.
    
    Args:
      camera_matrix: Camera calibration matrix
      dist_coeffs: Distortion coefficients
      pose_tracker: ArucoPoseTracker instance for marker detection
      finger_tracker: FingerArucoTracker instance for transformations
      marker_ids: List of ArUco marker IDs to detect
      marker_size_meters: Size of ArUco markers in meters
    """
    self.camera_matrix = camera_matrix
    self.dist_coeffs = dist_coeffs
    self.pose_tracker = pose_tracker
    self.finger_tracker = finger_tracker
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
      # TODO: Adapative filtering for light intensity
      success, image = cap.read()
      corners, ids, rejected = self.pose_tracker.detect_markers(image)
      if len(corners) > 0:
          detected_markers_image = self.pose_tracker.draw_detected_markers(image, corners, ids)
      else:
          detected_markers_image = image.copy()
      if not success: 
        print("Error reading from camera")
        return (2, None)
      
      # If we have a captured image, show it with confirmation options
      if captured_image is not None:
        corners, ids, rejected = self.pose_tracker.detect_markers(captured_image)
        if len(corners) > 0:
          try_detect_image = self.pose_tracker.draw_detected_markers(captured_image, corners, ids)
          cv.imshow("Markers found! If all four markers have been detected, press c to confirm or p to try again", try_detect_image)
        else:
          try_detect_image = captured_image.copy()
          cv.imshow("Markers not found! Press p to try again", try_detect_image)
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
        cv.imshow("Press p to capture or q to quit", detected_markers_image)
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('p'):
          captured_image = image.copy()  # Capture current frame
          cv.destroyWindow("Press p to capture or q to quit")
          gray = cv.cvtColor(captured_image, cv.COLOR_BGR2GRAY)
          adaptive_threshold_image = self.pose_tracker.apply_adaptive_threshold(gray)
          hist_eq_image = self.pose_tracker.apply_histogram_equalization(gray)

          cv.imshow("Gray", gray)
          cv.imshow("Adaptive threshold", adaptive_threshold_image)
          cv.imshow("Histogram equalization", hist_eq_image)
          cv.waitKey(0)
          cv.destroyAllWindows()

        elif key == ord('q'):
          cv.destroyAllWindows()
          return (2, None)

  def _detect_ridge_mask(self, image: np.ndarray, 
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
      poses = self.pose_tracker.get_marker_poses(image, marker_size_meters=MARKER_SIZE*IN_TO_METERS)
      success, marker_list_2d = self.finger_tracker.get_marker_polygon(
          self.marker_ids, poses, store_polygon=True
      )
      
      # Check if markers were found
      if not success or np.array_equal(marker_list_2d, [0, 0, 0, 0]):
          raise ValueError("ArUco markers not found in image!")
      
      print("ArUco markers found! Processing piano image...")
      
      # Transform to bird's eye view
      birdseye_image = self.finger_tracker.transform_image_to_birdseye(image, undistort=True)
      
      # Convert to grayscale
      gray = cv.cvtColor(birdseye_image, cv.COLOR_BGR2GRAY)
      
      # Apply median blur to reduce noise
      # Pad first to avoid edge artifacts
      pad_size = 2
      gray_padded = cv.copyMakeBorder(gray, pad_size, pad_size, pad_size, pad_size, 
                                      cv.BORDER_REPLICATE)
      med_image_padded = cv.medianBlur(gray_padded, median_blur_size)
      
      cv.imshow('med_image_padded', med_image_padded)
      cv.waitKey(0)
      cv.destroyAllWindows()

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
                          mask_crop_pct: float = 0.1) -> Tuple[np.ndarray, int]:
      """
      Crop the mask to remove ArUco marker regions from top and bottom.
      
      Args:
          mask: Input mask image
          mask_crop_pct: Percentage of image height to crop from top/bottom (default: 0.1)
          
      Returns:
          Tuple of (cropped_mask, y_offset) where y_offset is pixels cropped from top
      """
      if mask.dtype != np.uint8:
          mask = mask.astype(np.uint8)
      h, w = mask.shape
      
      # Calculate mask size for corners (only cropping top/bottom)
      mask_y = int(h * mask_crop_pct)
      
      cropped = mask[mask_y:h-mask_y, :].copy()
      
      return cropped, mask_y
  
  def _label_keys_from_boundary_mask(self, boundary_mask: np.ndarray,
                                start_midi: int,
                                expected_keys: int | None = None,
                                close_px: int = 3,
                                y_offset: int = 0,
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
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(boundary_mask, connectivity=8)
    
    # Filter out small noise regions (keep only regions above threshold)
    # Adjust min_area based on your noise level - smaller values keep more, larger removes more
    min_area = 120  # Minimum pixels to keep (removes isolated 1-2 pixel noise)
    walls_cleaned = np.zeros_like(boundary_mask)
    
    for label_id in range(1, num_labels):  # Skip background (label 0)
      area = stats[label_id, cv.CC_STAT_AREA]
      if area >= min_area:
        # Keep this region
        walls_cleaned[labels == label_id] = 255


    # Now proceed with closing operations to connect fragments
    # Skip opening - it breaks connections. Use closing to connect instead.
    walls_cleaned

    # 1) Vertical closing: connects broken vertical boundary segments
    # Use a tall vertical line kernel to connect vertical gaps (most important for piano keys)
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 5))  # 1 wide, 10 tall
    walls_vertical_closed = cv.morphologyEx(walls_cleaned, cv.MORPH_CLOSE, vertical_kernel)
    
    
    if debug_mode:
      cv.imshow('walls_vertical_closed', walls_vertical_closed)
      cv.waitKey(0)
      cv.destroyAllWindows()

    # 2) Horizontal closing: connects broken horizontal boundary segments
    # Use a WIDE horizontal line kernel to connect horizontal gaps (top/bottom edges)
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))  # 5 wide, 1 tall
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
        cnt = self.finger_tracker._map_coords_to_uncropped(cnt, y_offset)
        poly = self.finger_tracker._map_coords_to_uncropped(poly, y_offset)
        cx, cy = self.finger_tracker._map_coords_to_uncropped((cx, cy), y_offset)
        
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
    # TODO: Make this more robust by requesting orientation of piano
    keys.sort(key=lambda k: k['bbox'][0], reverse=True)
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
      
  
  def draw_labeled_keys(self, 
                        image: Union[np.ndarray, None], 
                        labeled: List[Dict], 
                        y_offset: int = 0) -> np.ndarray:
    """
    Draw labeled keys on the original camera image.
    
    Args:
        image: Original camera image (BGR or grayscale)
        labeled: List of key dictionaries from label_keys_from_boundary_mask
        y_offset: Y offset to add if keys are from cropped image (default: 0)
        
    Returns:
        Original image with drawn keys (BGR)
    """
    if image is None:
      return None
      
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
      image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    
    # Undistort the image to match the coordinate space of the transformed points
    if self.finger_tracker.new_camera_matrix is not None:
      image = cv.undistort(image, self.camera_matrix, self.dist_coeffs, None, self.finger_tracker.new_camera_matrix)
    else:
      image = cv.undistort(image, self.camera_matrix, self.dist_coeffs)
    # Create a copy to draw on
    result_image = image.copy()
    
    for p_key in labeled:
      cx, cy = p_key['centroid']
      contour = p_key['contour']

      # Transform from bird's-eye view to original image space
      cx, cy = self.finger_tracker.transform_point_from_birdseye_to_image((cx, cy))
      contour_original = self.finger_tracker.transform_contour_from_birdseye_to_image(contour)
      
      # Draw on original image
      result_image = cv.circle(result_image, (int(cx), int(cy)), 5, (255, 0, 127), -1)  # Magenta centroid
      result_image = cv.drawContours(result_image, [contour_original], -1, (0, 255, 0), 2)  # Green contour
      result_image = cv.putText(result_image, p_key['name'], (int(cx), int(cy-10)), 
        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return result_image
      
  
  def get_piano_calibration(self, 
          cap: cv.VideoCapture,
          start_midi: int = 48,
          mask_crop_pct: float = 0.09,
          debug_mode: bool = False) -> tuple[int, Dict]:
    """
    Complete pipeline: detect ridge mask, crop markers, label keys, and draw results.
    
    Args:
        start_midi: MIDI note of the left-most key (default: 48 for C3)
        mask_crop_pct: Percentage to crop from top/bottom for marker removal (default: 0.09)
        debug_mode: If True, display intermediate results (default: False)
        
    Returns:
        tuple of (status, dictionary) where:
            - status: 0 if successful, 1 if image confirmed, 2 if calibration cancelled
        Dictionary with keys:
            - 'gray': Grayscale bird's-eye view
            - 'ridge_mask': Binary ridge mask (before cropping)
            - 'cropped_mask': Ridge mask after cropping
            - 'y_offset': Y offset from cropping
            - 'labeled_keys': List of labeled key dictionaries
            - 'annotated_image': Image with keys drawn
    """
    while True:
      result, image = self._calibration_loop(cap)
      if result == 2:
        return result, None
      # Detect ridge mask
      gray, ridge_mask = self._detect_ridge_mask(image, debug_mode=debug_mode)
      

      # Crop marker regions
      cropped_mask, y_offset = self._crop_marker_regions(ridge_mask, mask_crop_pct)
      
      # Label keys in non-cropped image space
      labeled_keys = self._label_keys_from_boundary_mask(
          cropped_mask, start_midi, close_px=3, y_offset=y_offset, debug_mode=debug_mode
      )
      
      # Draw labeled keys on grayscale image
      annotated_image = self.draw_labeled_keys(image, labeled_keys, y_offset)
      # Show mask overlay for user confirmation
      
      print("\n Keyboard detected! Review the overlay:")

      print("  - Press 'c' to confirm and continue")
      print("  - Press 'q' to quit")
      print("  - Press 'r' to retry")
      
      cv.imshow('found piano', annotated_image)
      cv.waitKey(0)
      key = cv.waitKey(0) & 0xFF
      if key == ord('c'):
        cv.destroyAllWindows()
        break
      elif key == ord('q'):
        cv.destroyAllWindows()
        return 2, None
      elif key == ord('r'):
        cv.destroyAllWindows()
        continue

    return 0, {
      'labeled_keys': labeled_keys,
      'ridge_mask': ridge_mask,
      'cropped_mask': cropped_mask,
      'y_offset': y_offset
    }


def main():
  calib_npz = np.load(CAMERA_CALIBRATION_PATH)
  camera_matrix = calib_npz["camera_matrix"]
  dist_coeffs = calib_npz["dist_coeffs"]
  pose_tracker = ArucoPoseTracker(camera_matrix, dist_coeffs, mode=TrackingMode.STATIC, marker_ids=MARKER_IDS)
  finger_tracker = FingerArucoTracker(camera_matrix, dist_coeffs)
  marker_size_meters = MARKER_SIZE*IN_TO_METERS
  piano_calibrator = PianoCalibration(camera_matrix, dist_coeffs, pose_tracker, finger_tracker, MARKER_IDS, marker_size_meters)
  
  cap = cv.VideoCapture(0)
  result, piano_calibration_result = piano_calibrator.get_piano_calibration(cap)

  cv.waitKey(0)
  cv.destroyAllWindows()


if __name__ == "__main__":
    main()


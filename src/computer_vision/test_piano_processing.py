from cProfile import label
import cv2 as cv
import numpy as np
from typing import List, Dict, Tuple, Union
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.computer_vision.finger_aruco_tracker import FingerArucoTracker
from src.computer_vision.aruco_polygon_detector import ArucoPolygonDetector
from src.computer_vision.aruco_pose_tracker import ArucoPoseTracker
from src.core.constants import CAMERA_CALIBRATION_PATH, MARKER_IDS, MARKER_SIZE, IN_TO_METERS

# global variables
SAVE_PICTURES = False
LAST_MARKER_LIST_2D = None
  
try:
  print(f"Loading calibration file: {CAMERA_CALIBRATION_PATH}")
  calib_npz = np.load(CAMERA_CALIBRATION_PATH)
      
except(OSError): 
  raise Exception("Calibration file not found!")
except:
  raise Exception("Issue with reading calibration file")

camera_matrix = calib_npz["camera_matrix"]
dist_coeffs = calib_npz["dist_coeffs"]

pose_tracker = ArucoPoseTracker(camera_matrix, dist_coeffs, marker_ids=MARKER_IDS)
pose_tracker.configure_pose_filtering(
  enable_filtering=False,
  adaptive_thresholds=False,
  debug_output=False,
  enforce_z_axis_out=False,
  enable_moving_average=False
  )
finger_tracker = FingerArucoTracker()
polygon_detector = ArucoPolygonDetector(camera_matrix, dist_coeffs)


def calibration_loop(cap:cv.VideoCapture) -> tuple:
  """Calibration loop that allows capturing and confirming images.
  
  Returns:
    tuple: (status, image) where:
      - status 0: Continue loop
      - status 1: Image confirmed, proceed with processing
      - status 2: Quit calibration
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



def main():
  # Create output directory with timestamp
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

  if SAVE_PICTURES:
    output_dir = os.path.join(os.path.dirname(__file__), "calibration_output", f"calibration_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving calibration images to: {output_dir}")
  

    


  cap = cv.VideoCapture(0)

  print("Put the piano in frame and take a picture to calibrate.")
  print("Press p to capture frame, c to confirm, or q to quit.")
  
  # Get confirmed image from calibration loop
  result, image = calibration_loop(cap)
  
  if result == 2:
    print("Calibration cancelled by user")
    cap.release()
    cv.destroyAllWindows()
    return
  
  if result != 1:
    print("No image captured")
    cap.release()
    cv.destroyAllWindows()
    return
  
  print("Image confirmed! Running piano processing tests...")
  
  # Run adaptive and ridge tests on the confirmed image


  gray, ridge_mask = run_piano_processing_tests(image, camera_matrix, dist_coeffs, return_this='ridge')
  ridge_mask, ridge_y_offset = crop_marker_regions(ridge_mask, mask_margin_pct=0.09)

  labeled_ridge = label_keys_from_boundary_mask(ridge_mask,48)  

  ridge_image_contours = draw_labeled_keys(gray, labeled_ridge, y_offset=ridge_y_offset, show_contours=True)
  ridge_image_polygons = draw_labeled_keys(gray, labeled_ridge, y_offset=ridge_y_offset, show_polygons=True)

  print("Press any key to close windows...")
  cv.destroyAllWindows()

  ridge_image_contours = polygon_detector.transform_birdseye_to_image(ridge_image_contours)
  ridge_image_polygons = polygon_detector.transform_birdseye_to_image(ridge_image_polygons)

  
  cv.imshow("Ridge real space contours", ridge_image_contours)
  cv.imshow("Ridge real space polygons", ridge_image_polygons)
  print("Press any key to close windows...")
  cv.waitKey(0)
 
  cap.release()
  cv.destroyAllWindows()
  

  output_dir = os.path.join(os.path.dirname(__file__), "calibration_output", f"final_masks")

  cv.imwrite(os.path.join(output_dir, f"ridge_mask_{timestamp}.jpg"), ridge_mask)
  cv.imwrite(os.path.join(output_dir, f"ridge_image_contours_{timestamp}.jpg"), ridge_image_contours)
  cv.imwrite(os.path.join(output_dir, f"ridge_image_polygons_{timestamp}.jpg"), ridge_image_polygons)
  return 0



def run_piano_processing_tests(image, camera_matrix, dist_coeffs, 
                              return_this=None,output_dir=None)->Union[None,tuple[np.ndarray]]:
  """Run all piano processing tests on the confirmed image."""
  
  # Get AruCo marker poses and polygon
  poses = pose_tracker.get_marker_poses(image, MARKER_SIZE*IN_TO_METERS)
  marker_list_2d = polygon_detector.get_marker_polygon(MARKER_IDS, poses, store_polygon=True)
  global LAST_MARKER_LIST_2D
  LAST_MARKER_LIST_2D = marker_list_2d

  # Check if markers were found
  if np.array_equal(marker_list_2d, [0,0,0,0]):
    print("AruCo markers not found in image!")
    if output_dir is not None:
      cv.imwrite(os.path.join(output_dir, "00_no_markers_found.jpg"), image)
      cv.imshow("Original", image)
    return
  
  print("AruCo markers found! Processing piano image...")
  start_time = time.time()
  
  # Transform to bird's eye view
  warped = polygon_detector.transform_image_to_birdseye(image, marker_list_2d)
  transform_time = time.time() - start_time
  print(f"Image transformation (birdseye): {transform_time:.4f} seconds")

  # Convert to grayscale
  gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
  
  # Mask out ArUco marker regions by setting corners to white


  # Apply median blur to reduce noise
  # Pad first to avoid edge artifacts (especially important for ridge detection)
  pad_size = 2
  gray_padded = cv.copyMakeBorder(gray, pad_size, pad_size, pad_size, pad_size, 
                                  cv.BORDER_REPLICATE)
  med_image_padded = cv.medianBlur(gray_padded, 3)
  med_image = med_image_padded[pad_size:-pad_size, pad_size:-pad_size]

  # med_image = cv.dilate(med_image, cv.getStructuringElement(cv.MORPH_RECT, (3,3)), iterations=1)
  # Switch-like behavior using if/elif on return_this
  # Compute adaptive binary if needed
  binary = None
  adaptive_threshold_time = 0.0
  if return_this in ['binary', 'binary_inverted', 'adaptive', None]:
    start_time = time.time()
    binary = cv.adaptiveThreshold(med_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    adaptive_threshold_time = time.time() - start_time
    print(f"Adaptive threshold: {adaptive_threshold_time:.4f} seconds")

  if return_this == 'binary':
    return gray, binary

  if return_this == 'binary_inverted':
    return gray, cv.bitwise_not(binary)

  if return_this == 'adaptive':
    # Turn adaptive result into a boundary mask using morphological gradient
    inv = cv.bitwise_not(binary)
    grad = cv.morphologyEx(inv, cv.MORPH_GRADIENT, cv.getStructuringElement(cv.MORPH_RECT,(3,3)))
    _, boundary = cv.threshold(grad, 1, 255, cv.THRESH_BINARY)
    return gray, boundary

  if return_this == 'ridge':
    start_time = time.time()
    
    # Pad image to avoid edge artifacts in ridge detection
    pad_size = 5
    med_image_padded = cv.copyMakeBorder(med_image, pad_size, pad_size, pad_size, pad_size, 
                                         cv.BORDER_REPLICATE)
    
    ridge = cv.ximgproc.RidgeDetectionFilter_create()
    ridge_img_padded = ridge.getRidgeFilteredImage(med_image_padded)
    
    # Remove padding to get back to original size
    ridge_img = ridge_img_padded[pad_size:-pad_size, pad_size:-pad_size]
    
    ridge_time = time.time() - start_time
    print(f"Ridge detection: {ridge_time:.4f} seconds")
    _, ridge_bin = cv.threshold(ridge_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow('ridge otsu result', ridge_bin)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return gray, ridge_bin

  # Continue with full pipeline when no early return requested
  start_time = time.time()
  binary_inverted = cv.bitwise_not(binary)
  binary_invert_time = time.time() - start_time
  print(f"Binary inversion: {binary_invert_time:.4f} seconds")
  
  start_time = time.time()
  blurred = cv.GaussianBlur(gray, (3, 3), 0)
  gaussian_blur_time = time.time() - start_time
  print(f"Gaussian blur: {gaussian_blur_time:.4f} seconds")
  
  start_time = time.time()
  ridge = cv.ximgproc.RidgeDetectionFilter_create()
  ridge_image = ridge.getRidgeFilteredImage(gray)
  ridge_time = time.time() - start_time
  print(f"Ridge detection: {ridge_time:.4f} seconds")
 
  
  # Generate piano key masks from edge detection
  start_time = time.time()
  key_masks = generate_piano_key_masks(gray, binary_inverted)
  mask_generation_time = time.time() - start_time
  print(f"Key mask generation: {mask_generation_time:.4f} seconds")

  

  # Calculate total processing time
  total_processing_time = (transform_time + adaptive_threshold_time + 
                         binary_invert_time + gaussian_blur_time + 
                         ridge_time + mask_generation_time)
  print(f"\n=== TIMING SUMMARY ===")
  print(f"Total processing time: {total_processing_time:.4f} seconds")
  
  # Save images if enabled
  if output_dir is not None:
    save_processing_results(image, warped, gray, binary, binary_inverted, 
                          ridge_image, key_masks,
                          output_dir)
  
  # Display all processing steps
  display_processing_results(warped, binary_inverted, 
                            ridge_image, key_masks)
  
  # Wait for key press to keep windows open
  print("Press any key to close windows...")
  cv.waitKey(0)
  cv.destroyAllWindows()


def generate_piano_key_masks(gray_image, binary_image):
    """
    Generate piano key masks from edge-detected and binary images.
    
    Args:
        gray_image: Grayscale image of the piano
        binary_image: Binary thresholded image (keys should be white)
    
    Returns:
        dict: Dictionary containing various key masks and processing steps
    """
    height, width = gray_image.shape
    
    # Method 1: Morphological operations on binary image
    # Close gaps between key edges
    kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    closed_binary = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel_close)
    
    # Fill holes within keys
    kernel_fill = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    filled_binary = cv.morphologyEx(closed_binary, cv.MORPH_CLOSE, kernel_fill)
    
    
    # Method 3: Contour-based key detection
    contours, _ = cv.findContours(filled_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and aspect ratio (piano keys are roughly rectangular)
    key_contours = []
    min_area = (width * height) * 0.001  # Minimum 0.1% of image area
    max_area = (width * height) * 0.2  # Maximum 10% of image area
    
    for contour in contours:
        area = cv.contourArea(contour)
        if min_area < area < max_area:
            # Check aspect ratio (keys are wider than tall)
            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 0.5:  # Keys should be at least 0.5:1 ratio
                key_contours.append(contour)
    
    # Create contour-based mask
    contour_mask = np.zeros_like(gray_image)

    contour_mask = cv.drawContours(contour_mask, key_contours, -1, 255, -1)
    
    # Method 5: Template matching approach (simplified)
    # Create a simple rectangular template for white keys
    template_height = height // 8  # Approximate key height
    template_width = width // 52   # Approximate white key width (52 white keys on piano)
    
    # Simple rectangular template
    template = np.ones((template_height, template_width), dtype=np.uint8) * 255
    
    # Match template
    result = cv.matchTemplate(filled_binary, template, cv.TM_CCOEFF_NORMED)
    locations = np.where(result >= 0.3)  # Threshold for matches
    
    # Create template-based mask
    template_mask = np.zeros_like(gray_image)
    for pt in zip(*locations[::-1]):
        cv.rectangle(template_mask, pt, (pt[0] + template_width, pt[1] + template_height), 255, -1)
    
    # Method 6: Adaptive thresholding with different parameters
    # Try different adaptive threshold parameters
    adaptive_masks = {}
    for block_size in [11, 15, 21]:
        for c in [2, 5, 10]:
            adaptive_mask = cv.adaptiveThreshold(
                gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv.THRESH_BINARY, block_size, c
            )
            adaptive_masks[f'adaptive_{block_size}_{c}'] = adaptive_mask
    
    # Combine best methods
    # Weighted combination of different approaches
    combined_mask = cv.addWeighted(filled_binary, 0.4, contour_mask, 0.3, 0)
    
    # Final cleanup
    kernel_clean = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    final_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel_clean)
    final_mask = cv.morphologyEx(final_mask, cv.MORPH_OPEN, kernel_clean)
    
    return {
        'binary_filled': filled_binary,
        'contour_mask': contour_mask,
        'template_mask': template_mask,
        'combined_mask': combined_mask,
        'final_mask': final_mask,
        'adaptive_masks': adaptive_masks,
        'key_contours': key_contours,
        'num_keys_detected': len(key_contours)
    }


def crop_marker_regions(mask: np.ndarray, mask_margin_pct: float = 0.1) -> Tuple[np.ndarray, int]:
    """
    This will crop the mask to remove the aruco markers
    
    Returns:
        Tuple of (cropped_mask, y_offset) where y_offset is the pixels cropped from top
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    h, w = mask.shape
    
    # Calculate mask size for corners (only cropping top/bottom)
    mask_y = int(h * mask_margin_pct)

    cropped = mask[mask_y:h-mask_y, :].copy()

    return cropped, mask_y


def map_coords_to_original(coords_or_contour, y_offset: int):
    """
    Map coordinates or contours from cropped image space back to original image space.
    
    Args:
        coords_or_contour: Can be:
            - Tuple (x, y) for a single point
            - NumPy array with shape (N, 1, 2) for contour/polygon (from cv.findContours, cv.approxPolyDP)
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


def label_keys_from_boundary_mask(boundary_mask: np.ndarray,
                                  start_midi: int,
                                  expected_keys: int | None = None,
                                  close_px: int = 3) -> List[Dict]:
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
  
  cv.imshow('walls_vertical_and_horizontal', walls_connected)
  cv.waitKey(0)
  cv.destroyAllWindows()

  # 3) General closing: connects any remaining nearby fragments
  # This catches diagonal connections and general gaps
  # close_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
  # walls_connected = cv.morphologyEx(walls_connected, cv.MORPH_CLOSE, close_kernel)
  
  # cv.imshow('walls_connected_and_square_kernel', walls_connected)
  # cv.waitKey(0)
  # cv.destroyAllWindows()

  
 
  


  # 4) Invert to get key regions as 1s
  regions = cv.bitwise_not(walls_connected)
  regions = (regions > 0).astype(np.uint8)
  
  # 5) Opening on key regions: removes small artifacts inside keys (white specks)
  # This cleans up noise that appears as small white dots in the black key regions
  artifact_removal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
  regions = cv.morphologyEx(regions, cv.MORPH_OPEN, artifact_removal_kernel, iterations=1)
  

  # 6) Slight dilation to recover area lost during wall dilation
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
      
   

    

def draw_labeled_keys(image:np.ndarray, labeled:List[dict], y_offset: int = 0, show_contours: bool = False, show_polygons: bool = True):
  """Draw labeled keys on the image, mapping from cropped space to original if needed.
  
  Args:
    image: Original (uncropped) image to draw on
    labeled: List of key dictionaries from label_keys_from_boundary_mask
    y_offset: Y offset to add if keys are from cropped image (default: 0)
    show_contours: If True, draw contours on the image
    show_polygons: If True, draw polygons on the image
  """
  if len(image.shape) == 2:
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    
  for p_key in labeled:
    # Map bbox coordinates back to original image space
    x, y, w, h = map_coords_to_original(p_key['bbox'], y_offset)
    pt1 = (int(x), int(y))
    pt2 = (int(x+w), int(y+h))
    
    # Map centroid back to original image space
    cx, cy = map_coords_to_original(p_key['centroid'], y_offset)
    
    # Map polygon back to original image space
    poly_original = map_coords_to_original(p_key['poly'], y_offset)
    contour_original = map_coords_to_original(p_key['contour'], y_offset)

    # Draw on original image
    image = cv.circle(image, (cx, cy), 5, (255, 0, 127), -1)  # Magenta centroid
    image = cv.putText(image, p_key['name'], (cx, cy-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    if show_polygons:
      image = cv.polylines(image, [poly_original], isClosed=True, color=(139, 0, 139), thickness=2)  # Dark pink (BGR)
    if show_contours:
      image = cv.drawContours(image, [contour_original], -1, (0, 255, 0) if p_key['is_black'] else (0, 0, 255), 2)  # Green or Red contour
  return image


def save_processing_results(image, warped, gray, binary, binary_inverted, 
                           ridge_image, 
                           key_masks, output_dir):
  """Save all processing results to files."""
  start_time = time.time()
  
  # Save basic processing steps
  cv.imwrite(os.path.join(output_dir, "01_original.jpg"), image)
  cv.imwrite(os.path.join(output_dir, "02_warped_birdseye.jpg"), warped)
  cv.imwrite(os.path.join(output_dir, "03_grayscale.jpg"), gray)
  cv.imwrite(os.path.join(output_dir, "04_adaptive_binary.jpg"), binary)
  cv.imwrite(os.path.join(output_dir, "05_binary_inverted.jpg"), binary_inverted)
  
  # Save edge detection results
  cv.imwrite(os.path.join(output_dir, "16_ridge_detection.jpg"), ridge_image)
  
  # Save key masks
  cv.imwrite(os.path.join(output_dir, "17_binary_filled.jpg"), key_masks['binary_filled'])
  cv.imwrite(os.path.join(output_dir, "19_contour_mask.jpg"), key_masks['contour_mask'])
  cv.imwrite(os.path.join(output_dir, "20_template_mask.jpg"), key_masks['template_mask'])
  cv.imwrite(os.path.join(output_dir, "21_combined_mask.jpg"), key_masks['combined_mask'])
  cv.imwrite(os.path.join(output_dir, "22_final_mask.jpg"), key_masks['final_mask'])

  # Save adaptive threshold results
  for name, mask in key_masks['adaptive_masks'].items():
    cv.imwrite(os.path.join(output_dir, f"25_{name}.jpg"), mask)
  
  save_time = time.time() - start_time
  print(f"Image saving operations: {save_time:.4f} seconds")
  print(f"Saved calibration images to {output_dir}")
  print(f"Detected {key_masks['num_keys_detected']} potential piano keys")
    

def display_processing_results(warped, binary_inverted, 
                             ridge_image, key_masks):
  """Display all processing results in windows."""
  
  # Display basic processing steps
  cv.imshow("Warped (Bird's Eye)", warped)
  cv.imshow("Binary Inverted", binary_inverted)
  cv.imshow("Ridge", ridge_image)
  
  # Display key masks
  cv.imshow("Mask Binary Filled", key_masks['binary_filled'])
  cv.imshow("Mask Contour Mask", key_masks['contour_mask'])
  cv.imshow("Mask Combined Mask", key_masks['combined_mask'])
  cv.imshow("Mask Final Mask", key_masks['final_mask'])
  cv.imshow("template Mask", key_masks['template_mask'])

  print(f"\nKey Detection Results:")
  print(f"Number of keys detected: {key_masks['num_keys_detected']}")
  print(f"Expected keys on piano: ~52 white keys + ~36 black keys = ~88 total")
if __name__ == '__main__':
  main()
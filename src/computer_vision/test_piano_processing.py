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

SAVE_PICTURES = False


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
  if SAVE_PICTURES:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "calibration_output", f"calibration_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving calibration images to: {output_dir}")
  
  try:
    calib_npz = np.load(CAMERA_CALIBRATION_PATH)
        
  except(OSError): 
    print("Calibration file not found!")
    return
  except:
    print("Issue with reading calibration file")
    return
    
  camera_matrix = calib_npz["camera_matrix"]
  dist_coeffs = calib_npz["dist_coeffs"]

  pose_tracker = ArucoPoseTracker(marker_ids=MARKER_IDS)
  pose_tracker.configure_pose_filtering(
    enable_filtering=False,
    adaptive_thresholds=False,
    debug_output=False,
    enforce_z_axis_out=False,
    enable_moving_average=False
    )
  finger_tracker = FingerArucoTracker()
  polygon_detector = ArucoPolygonDetector(camera_matrix, dist_coeffs)

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
  
  # Run all processing tests on the confirmed image
  gray, adaptive_mask = run_piano_processing_tests(image, camera_matrix, dist_coeffs, pose_tracker, 
            finger_tracker, polygon_detector, return_this='adaptive')
  gray, ridge_mask = run_piano_processing_tests(image, camera_matrix, dist_coeffs, pose_tracker, 
      finger_tracker, polygon_detector, return_this='ridge')
  labeled_adaptive = label_keys_from_boundary_mask(adaptive_mask,48)
  labeled_ridge = label_keys_from_boundary_mask(ridge_mask,48)

  adaptive_image = draw_labeled_keys(gray, labeled_adaptive)
  ridge_image = draw_labeled_keys(gray, labeled_ridge)

  cv.imshow("Adaptive", adaptive_image)
  cv.imshow("Ridge", ridge_image)
  cv.waitKey(0)
  
  cap.release()
  cv.destroyAllWindows()




def run_piano_processing_tests(image, camera_matrix, dist_coeffs, pose_tracker, 
                              finger_tracker, polygon_detector, return_this=None,output_dir=None)->Union[None,tuple[np.ndarray]]:
  """Run all piano processing tests on the confirmed image."""
  
  # Get AruCo marker poses and polygon
  poses = pose_tracker.get_marker_poses(image, camera_matrix, dist_coeffs, MARKER_SIZE*IN_TO_METERS)
  marker_list_2d = polygon_detector.get_marker_polygon(MARKER_IDS, poses, image, MARKER_SIZE * IN_TO_METERS)
  
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
  warped = finger_tracker.transform_image_to_birdseye(image, marker_list_2d)
  transform_time = time.time() - start_time
  print(f"Image transformation (birdseye): {transform_time:.4f} seconds")

  # Convert to grayscale
  gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

  # Apply median blur to reduce noise
  med_image = cv.medianBlur(gray, 3)

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
    cv.imshow('grad result', grad)
    cv.waitKey(0)
    cv.destroyAllWindows()
    _, boundary = cv.threshold(grad, 1, 255, cv.THRESH_BINARY)
    return gray, boundary

  if return_this == 'ridge':
    start_time = time.time()
    ridge = cv.ximgproc.RidgeDetectionFilter_create()
    ridge_img = ridge.getRidgeFilteredImage(gray)
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




def label_keys_from_boundary_mask(boundary_mask: np.ndarray,
                                  start_midi: int,
                                  expected_keys: int | None = None,
                                  close_px: int = 3) -> List[Dict]:
    """
    boundary_mask: uint8 image, 255 at key boundaries, 0 elsewhere
    start_midi: MIDI note of the left-most key in view (e.g., 21 for A0, 48 for C3)
    expected_keys: optional sanity check on count
    close_px: thickness to ensure separators are closed

    returns list of dicts: [{name, midi, is_black, bbox, contour, centroid}, ...]
    """

    h, w = boundary_mask.shape[:2]

    # 1) Thicken boundaries to close small gaps
    k = cv.getStructuringElement(cv.MORPH_RECT, (close_px, close_px))
    walls = cv.dilate(boundary_mask, k, iterations=1)
    
    # 2) Invert to get key regions as 1s
    regions = cv.bitwise_not(walls)
    regions = (regions > 0).astype(np.uint8)
    # cv.imshow('regions pre morph', regions)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Optional: remove outer background by keeping largest inside area
    # Fill holes so each key region is solid
    regions = cv.morphologyEx(regions, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT,(3,3)))
  
  
    # 3) Connected components
    num, labels, stats, cents = cv.connectedComponentsWithStats(regions, connectivity=4, ltype=cv.CV_32S)

    # Collect candidates (skip label 0 = background)
    comps = []
    for lbl in range(1, num):
        x, y, ww, hh, area = stats[lbl]
        if area < 0.001*w*h:  # filter tiny specks
            continue
        cx, cy = cents[lbl]
        aspect = ww / max(1, hh)
        comps.append({
            'label': lbl,
            'bbox': (x, y, ww, hh),
            'centroid': (float(cx), float(cy)),
            'area': int(area),
            'aspect': float(aspect)
        })

    # 4) Classify black vs white keys (black keys are shorter and near top)
    # Heuristic thresholds; tune to your birdseye scale
    print(comps)
    if not comps:
        return []

    heights = np.array([c['bbox'][3] for c in comps], dtype=float)
    median_h = float(np.median(heights))
    for c in comps:
        _, y, _, hh = c['bbox']
        # black keys typically have ~50-70% of white-key height and sit higher (smaller y)
        c['is_black'] = hh < 0.75*median_h and y < 0.35*h

    # 5) Sort by x (left-to-right). Black keys will ride on top in naming if needed later
    comps.sort(key=lambda c: c['centroid'][0])

    # 6) Assign names from MIDI sequence
    # 12-semitone cycle; map MIDI -> name
    NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    def name_from_midi(m: int) -> Tuple[str,bool]:
        n = NAMES[m % 12]
        octave = (m // 12) - 1
        return f"{n}{octave}", ('#' in n)

    if expected_keys is not None and len(comps) != expected_keys:
        # You can log or adjust later; we still proceed
        pass

    labeled = []
    midi = start_midi
    for c in comps:
        name, is_black_from_midi = name_from_midi(midi)
        # If the geometry classification disagrees with theory, you may skip or adjust midi here.
        labeled.append({
            'name': name,
            'midi': midi,
            'is_black': c['is_black'],
            'bbox': c['bbox'],
            'centroid': c['centroid'],
            'label': c['label']
        })
        midi += 1

    return labeled

def draw_labeled_keys(image:np.ndarray, labeled:List[dict]):
  """Draw labeled keys on the image."""
  if len(image.shape) == 2:
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
  for p_key in labeled:
    x,y,w,h = p_key['bbox']
    cx,cy = map(int, p_key['centroid'])
    
    # points for rectangle
    pt1 = (int(x), int(y))
    pt2 = (int(x+w), int(y+h))
    image = cv.circle(image, (cx, cy), 5, (255, 0, 127), -1)    
    image = cv.putText(image, p_key['name'], (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    image = cv.rectangle(image, pt1, pt2, (0, 0, 255), 2)
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
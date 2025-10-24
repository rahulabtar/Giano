import cv2 as cv
import numpy as np
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

def analyze_exposure_quality(image):
    """Analyze image quality for exposure optimization"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Calculate statistics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # Count overexposed pixels (too bright)
    overexposed_pixels = np.sum(gray > 240)
    total_pixels = gray.size
    overexposed_ratio = overexposed_pixels / total_pixels
    
    # Count underexposed pixels (too dark)
    underexposed_pixels = np.sum(gray < 15)
    underexposed_ratio = underexposed_pixels / total_pixels
    
    # Ideal brightness range is around 100-150
    brightness_score = 1.0 - abs(mean_brightness - 125) / 125
    
    # Penalize overexposed images heavily
    overexposure_penalty = overexposed_ratio * 2
    
    # Calculate overall quality score
    quality_score = brightness_score - overexposure_penalty - (underexposed_ratio * 0.5)
    
    return {
        'mean_brightness': mean_brightness,
        'std_brightness': std_brightness,
        'overexposed_ratio': overexposed_ratio,
        'underexposed_ratio': underexposed_ratio,
        'quality_score': quality_score
    }

def generate_piano_key_masks(gray_image, edges_image, binary_image):
    """
    Generate piano key masks from edge-detected and binary images.
    
    Args:
        gray_image: Grayscale image of the piano
        edges_image: Edge-detected image (Canny edges)
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
    
    # Method 2: Watershed segmentation on edges
    # Create distance transform
    dist_transform = cv.distanceTransform(filled_binary, cv.DIST_L2, 5)
    
    # Find local maxima as key centers
    _, sure_fg = cv.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Create markers for watershed
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1  # Add 1 to all labels so that sure background is not 0, but 1
    
    # Mark the region of unknown with zero
    markers[filled_binary == 0] = 0
    
    # Apply watershed
    markers = cv.watershed(cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR), markers)
    
    # Create mask from watershed result
    watershed_mask = np.zeros_like(gray_image)
    watershed_mask[markers > 1] = 255
    
    # Method 3: Contour-based key detection
    contours, _ = cv.findContours(filled_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and aspect ratio (piano keys are roughly rectangular)
    key_contours = []
    min_area = (width * height) * 0.001  # Minimum 0.1% of image area
    max_area = (width * height) * 0.1   # Maximum 10% of image area
    
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
    cv.drawContours(contour_mask, key_contours, -1, 255, -1)
    
    # Method 4: Horizontal line detection for key separations
    # Detect horizontal lines that separate keys
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (width//4, 1))
    horizontal_lines = cv.morphologyEx(edges_image, cv.MORPH_OPEN, horizontal_kernel)
    
    # Detect vertical lines for key boundaries
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, height//8))
    vertical_lines = cv.morphologyEx(edges_image, cv.MORPH_OPEN, vertical_kernel)
    
    # Combine horizontal and vertical lines
    combined_lines = cv.add(horizontal_lines, vertical_lines)
    
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
    combined_mask = cv.addWeighted(filled_binary, 0.4, watershed_mask, 0.3, 0)
    combined_mask = cv.addWeighted(combined_mask, 1.0, contour_mask, 0.3, 0)
    
    # Final cleanup
    kernel_clean = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    final_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel_clean)
    final_mask = cv.morphologyEx(final_mask, cv.MORPH_OPEN, kernel_clean)
    
    return {
        'binary_filled': filled_binary,
        'watershed_mask': watershed_mask,
        'contour_mask': contour_mask,
        'template_mask': template_mask,
        'combined_mask': combined_mask,
        'final_mask': final_mask,
        'horizontal_lines': horizontal_lines,
        'vertical_lines': vertical_lines,
        'combined_lines': combined_lines,
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
  run_piano_processing_tests(image, camera_matrix, dist_coeffs, pose_tracker, 
                           finger_tracker, polygon_detector)
  
  cap.release()
  cv.destroyAllWindows()




def run_piano_processing_tests(image, camera_matrix, dist_coeffs, pose_tracker, 
                              finger_tracker, polygon_detector, output_dir=None):
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

  
  # Apply adaptive threshold
  start_time = time.time()
  binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
  adaptive_threshold_time = time.time() - start_time
  print(f"Adaptive threshold: {adaptive_threshold_time:.4f} seconds")
  
  # Invert binary image so keys are white
  start_time = time.time()
  binary_inverted = cv.bitwise_not(binary)
  binary_invert_time = time.time() - start_time
  print(f"Binary inversion: {binary_invert_time:.4f} seconds")
  
  # Apply Gaussian blur for edge detection
  start_time = time.time()
  blurred = cv.GaussianBlur(gray, (3, 3), 0)
  gaussian_blur_time = time.time() - start_time
  print(f"Gaussian blur: {gaussian_blur_time:.4f} seconds")
  
  # Try multiple edge detection approaches
  start_time = time.time()
  edges1 = cv.Canny(blurred, 30, 100)  # Lower thresholds
  edges2 = cv.Canny(blurred, 50, 150)  # Original
  edges3 = cv.Canny(blurred, 20, 60)   # Very sensitive
  canny_edges_time = time.time() - start_time
  print(f"Canny edge detection (3 variants): {canny_edges_time:.4f} seconds")

  # Ridge detection filter
  start_time = time.time()
  ridge = cv.ximgproc.RidgeDetectionFilter_create()
  ridge_image = ridge.getRidgeFilteredImage(gray)
  ridge_time = time.time() - start_time
  print(f"Ridge detection: {ridge_time:.4f} seconds")
  
  # Sobel edge detection
  start_time = time.time()
  sobel_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
  sobel_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
  sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
  sobel_edges = np.uint8(sobel_edges / sobel_edges.max() * 255)
  sobel_time = time.time() - start_time
  print(f"Sobel edge detection: {sobel_time:.4f} seconds")
  
  # Generate piano key masks from edge detection
  start_time = time.time()
  key_masks = generate_piano_key_masks(gray, edges2, binary_inverted)
  mask_generation_time = time.time() - start_time
  print(f"Key mask generation: {mask_generation_time:.4f} seconds")
  
  # Calculate total processing time
  total_processing_time = (transform_time + adaptive_threshold_time + 
                         binary_invert_time + gaussian_blur_time + canny_edges_time + 
                         ridge_time + sobel_time + mask_generation_time)
  print(f"\n=== TIMING SUMMARY ===")
  print(f"Total processing time: {total_processing_time:.4f} seconds")
  
  # Save images if enabled
  if output_dir is not None:
    save_processing_results(image, warped, gray, binary, binary_inverted, 
                          edges1, edges2, edges3, sobel_edges, ridge_image, 
                          key_masks, output_dir)
  
  # Display all processing steps
  display_processing_results(warped, binary_inverted, edges2, sobel_edges, 
                            ridge_image, key_masks)
  
  # Wait for key press to keep windows open
  print("Press any key to close windows...")
  cv.waitKey(0)
  cv.destroyAllWindows()


def save_processing_results(image, warped, gray, binary, binary_inverted, 
                           edges1, edges2, edges3, sobel_edges, ridge_image, 
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
  cv.imwrite(os.path.join(output_dir, "12_edges_canny_30_100.jpg"), edges1)
  cv.imwrite(os.path.join(output_dir, "13_edges_canny_50_150.jpg"), edges2)
  cv.imwrite(os.path.join(output_dir, "14_edges_canny_20_60.jpg"), edges3)
  cv.imwrite(os.path.join(output_dir, "15_edges_sobel.jpg"), sobel_edges)
  cv.imwrite(os.path.join(output_dir, "16_ridge_detection.jpg"), ridge_image)
  
  # Save key masks
  cv.imwrite(os.path.join(output_dir, "17_binary_filled.jpg"), key_masks['binary_filled'])
  cv.imwrite(os.path.join(output_dir, "18_watershed_mask.jpg"), key_masks['watershed_mask'])
  cv.imwrite(os.path.join(output_dir, "19_contour_mask.jpg"), key_masks['contour_mask'])
  cv.imwrite(os.path.join(output_dir, "20_template_mask.jpg"), key_masks['template_mask'])
  cv.imwrite(os.path.join(output_dir, "21_combined_mask.jpg"), key_masks['combined_mask'])
  cv.imwrite(os.path.join(output_dir, "22_final_mask.jpg"), key_masks['final_mask'])
  cv.imwrite(os.path.join(output_dir, "23_horizontal_lines.jpg"), key_masks['horizontal_lines'])
  cv.imwrite(os.path.join(output_dir, "24_vertical_lines.jpg"), key_masks['vertical_lines'])
  
  # Save adaptive threshold results
  for name, mask in key_masks['adaptive_masks'].items():
    cv.imwrite(os.path.join(output_dir, f"25_{name}.jpg"), mask)
  
  save_time = time.time() - start_time
  print(f"Image saving operations: {save_time:.4f} seconds")
  print(f"Saved calibration images to {output_dir}")
  print(f"Detected {key_masks['num_keys_detected']} potential piano keys")


def display_processing_results(warped, binary_inverted, edges2, sobel_edges, 
                             ridge_image, key_masks):
  """Display all processing results in windows."""
  
  # Display basic processing steps
  cv.imshow("Warped (Bird's Eye)", warped)
  cv.imshow("Binary Inverted", binary_inverted)
  cv.imshow("Canny Edges", edges2)
  cv.imshow("Sobel edges", sobel_edges)
  cv.imshow("Ridge", ridge_image)
  
  # Display key masks
  cv.imshow("Binary Filled", key_masks['binary_filled'])
  cv.imshow("Watershed Mask", key_masks['watershed_mask'])
  cv.imshow("Contour Mask", key_masks['contour_mask'])
  cv.imshow("Combined Mask", key_masks['combined_mask'])
  cv.imshow("Final Mask", key_masks['final_mask'])
  cv.imshow("Horizontal Lines", key_masks['horizontal_lines'])
  cv.imshow("Vertical Lines", key_masks['vertical_lines'])
  
  print(f"\nKey Detection Results:")
  print(f"Number of keys detected: {key_masks['num_keys_detected']}")
  print(f"Expected keys on piano: ~52 white keys + ~36 black keys = ~88 total")
if __name__ == '__main__':
  main()
import cv2 as cv
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.computer_vision.finger_aruco_tracker import FingerArucoTracker
from src.computer_vision.aruco_polygon_detector import ArucoPolygonDetector
from src.computer_vision.aruco_pose_tracker import ArucoPoseTracker
from src.core.constants import CAMERA_CALIBRATION_PATH, MARKER_IDS, MARKER_SIZE, IN_TO_METERS

SAVE_PICTURES = False

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
  except:
    print("Issue with reading calibration file")
    
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
  success, image = cap.read()
  if not success:
    raise("Camera taking picture error")

  # Get AruCo marker poses and polygon
  poses = pose_tracker.get_marker_poses(image, camera_matrix, dist_coeffs, MARKER_SIZE*IN_TO_METERS)
  marker_list_2d = polygon_detector.get_marker_polygon(MARKER_IDS, poses, image, MARKER_SIZE * IN_TO_METERS)
  
  if not np.array_equal(marker_list_2d, [0,0,0,0]):
    warped = finger_tracker.transform_image_to_birdseye(image, marker_list_2d)
    
    
    gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    
    # Invert binary image so keys are white
    binary_inverted = cv.bitwise_not(binary)
    
    # Try edge detection with different parameters
    # Apply Gaussian blur first to reduce noise
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    
    # Try multiple edge detection approaches
    edges1 = cv.Canny(blurred, 30, 100)  # Lower thresholds
    edges2 = cv.Canny(blurred, 50, 150)  # Original
    edges3 = cv.Canny(blurred, 20, 60)   # Very sensitive
    
    # Try Sobel edge detection as alternative
    sobel_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_edges = np.uint8(sobel_edges / sobel_edges.max() * 255)
    
   
    
      
    if SAVE_PICTURES: 
      # Save all processing steps to files
      cv.imwrite(os.path.join(output_dir, "01_original.jpg"), image)
      cv.imwrite(os.path.join(output_dir, "02_warped_birdseye.jpg"), warped)
      cv.imwrite(os.path.join(output_dir, "03_grayscale.jpg"), gray)
      cv.imwrite(os.path.join(output_dir, "04_adaptive_binary.jpg"), binary)
      cv.imwrite(os.path.join(output_dir, "05_binary_inverted.jpg"), binary_inverted)
      
      # Save multiple threshold results
      cv.imwrite(os.path.join(output_dir, "06_otsu_binary.jpg"), otsu_binary)
      cv.imwrite(os.path.join(output_dir, "07_simple_binary_200.jpg"), simple_binary_200)
      cv.imwrite(os.path.join(output_dir, "08_simple_binary_150.jpg"), simple_binary_150)
      cv.imwrite(os.path.join(output_dir, "09_simple_binary_100.jpg"), simple_binary_100)
      
      # Save multiple edge detection results
      cv.imwrite(os.path.join(output_dir, "12_edges_canny_30_100.jpg"), edges1)
      cv.imwrite(os.path.join(output_dir, "13_edges_canny_50_150.jpg"), edges2)
      cv.imwrite(os.path.join(output_dir, "14_edges_canny_20_60.jpg"), edges3)
      cv.imwrite(os.path.join(output_dir, "15_edges_sobel.jpg"), sobel_edges)
      print(f"Saved calibration images to {output_dir}")
    
    # Display all processing steps
    # cv.imshow("Warped (Bird's Eye)", warped)
    # cv.imshow("Grayscale", gray)
    # cv.imshow("Adaptive Binary", binary)
    # cv.imshow("Binary Inverted", binary_inverted)
    cv.imshow("Simple Binary 160", simple_binary_150)
    # cv.imshow("Edges 1", edges1)
    # cv.imshow("Edges 2", edges2)
    # cv.imshow("Edges 3", edges3)
    # cv.imshow("Blurred", blurred)

    
  else:
    print("AruCo markers not found")
    cv.imwrite(os.path.join(output_dir, "00_no_markers_found.jpg"), image)
    cv.imshow("Original", image)
  
  # Wait for key press to keep windows open
  print("Press any key to close windows...")
  cv.waitKey(0)
  cv.destroyAllWindows()

    


if __name__ == '__main__':
  main()
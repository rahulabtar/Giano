import cv2 as cv
import numpy as np
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.constants import CAMERA_CALIBRATION_PATH, DEFAULT_CAMERA_ID

def main():
    """Display live camera feed with distortion correction applied."""
    
    # Load camera calibration data
    try:
        calib_data = np.load(CAMERA_CALIBRATION_PATH)
        camera_matrix = calib_data["camera_matrix"]
        dist_coeffs = calib_data["dist_coeffs"]
        print(f"Loaded calibration from: {CAMERA_CALIBRATION_PATH}")
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients: {dist_coeffs}")
    except FileNotFoundError:
        print(f"Error: Calibration file not found at {CAMERA_CALIBRATION_PATH}")
        print("Please run camera calibration first!")
        return
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return
    
    # Open camera
    cap = cv.VideoCapture(DEFAULT_CAMERA_ID+1)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {DEFAULT_CAMERA_ID+1}")
        return
    
    # Get camera resolution
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    
    # Get optimal new camera matrix for undistortion
    # This can improve the undistorted image quality by cropping/zooming
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )
    
    print("\nControls:")
    print("  Press 'q' to quit")
    print("  Press 's' to save current frame")
    print("  Press 'f' to toggle full undistortion (no cropping)")
    
    show_full = False  # Toggle between cropped and full undistortion
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Apply distortion correction
        if show_full:
            # Full undistortion (may show black borders)
            undistorted = cv.undistort(frame, camera_matrix, dist_coeffs)
        else:
            # Cropped undistortion (removes black borders, uses optimal matrix)
            undistorted = cv.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
            # Crop to ROI if needed
            x, y, w, h = roi
            if w > 0 and h > 0:
                undistorted = undistorted[y:y+h, x:x+w]
        
        # Resize if needed to fit on screen (optional)
        # You can comment this out if you want full resolution
        display_scale = 0.7  # Scale down for display
        if display_scale < 1.0:
            orig_display = cv.resize(frame, None, fx=display_scale, fy=display_scale)
            undist_display = cv.resize(undistorted, None, fx=display_scale, fy=display_scale)
        else:
            orig_display = frame
            undist_display = undistorted
        
        # Create side-by-side comparison
        h1, w1 = orig_display.shape[:2]
        h2, w2 = undist_display.shape[:2]
        
        # Make heights match
        if h1 != h2:
            if h1 > h2:
                undist_display = cv.resize(undist_display, (w2, h1))
            else:
                orig_display = cv.resize(orig_display, (w1, h2))
        
        combined = np.hstack([orig_display, undist_display])
        
        # Add labels
        cv.putText(combined, "Original (Distorted)", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(combined, "Undistorted", (w1 + 10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add mode indicator
        mode_text = "Full (with borders)" if show_full else "Cropped (optimal)"
        cv.putText(combined, f"Mode: {mode_text}", (10, combined.shape[0] - 10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv.imshow("Live Camera - Original vs Undistorted", combined)
        
        # Handle keyboard input
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frames
            cv.imwrite(f"original_frame_{frame_count}.jpg", frame)
            cv.imwrite(f"undistorted_frame_{frame_count}.jpg", undistorted)
            print(f"Saved frames: original_frame_{frame_count}.jpg, undistorted_frame_{frame_count}.jpg")
            frame_count += 1
        elif key == ord('f'):
            show_full = not show_full
            print(f"Switched to {'full' if show_full else 'cropped'} undistortion mode")
    
    cap.release()
    cv.destroyAllWindows()
    print("Camera feed closed.")


if __name__ == '__main__':
    main()


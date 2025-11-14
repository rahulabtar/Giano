import cv2 as cv
import numpy as np
import glob
import os
import sys
from datetime import datetime

# Add project root to Python path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.constants import ASSETS_DIR

PATTERNS = {"chessboard_9x6": "https://github.com/opencv/opencv/blob/master/doc/pattern.png"}
CALIBRATION_DIR = os.path.join(ASSETS_DIR, "calibration")
CALIBRATION_IMAGE_DIR = os.path.join(ASSETS_DIR, "calibration_images")

def calibrate_camera(user_filename:str):
    """
    Calibrate camera using chessboard pattern.
    Print out a chessboard pattern and take 10-20 photos from different angles.
    """
    # Chessboard dimensions (internal corners)
    CHECKERBOARD = (9, 6)  # 9x6 internal corners (10x7 squares)
    
    # Prepare object points (3D points in real world space)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Take photos with your camera first, then load them
    images = glob.glob(os.path.join(CALIBRATION_IMAGE_DIR, '*.jpg'))  # Put your calibration photos here
    
    if not images:
        print("No calibration images found! Take 15-20 photos of a chessboard pattern.")
        return None, None
    
    for fname in images:
        print("filename: ", fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Draw and display corners (optional)
            cv.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
            cv.imshow('Chessboard', img)
            cv.waitKey(100)
    
    cv.destroyAllWindows()
    
    if len(objpoints) < 10:
        print(f"Only found {len(objpoints)} good images. Need at least 10 for good calibration.")
        return None, None
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    print("Camera calibration completed!")
    print(f"Camera Matrix:\n{camera_matrix}")
    print(f"Distortion Coefficients:\n{dist_coeffs}")
    print(f"Reprojection Error: {ret}")
    
    # Save calibration data
    np.savez(os.path.join(CALIBRATION_IMAGE_DIR, user_filename, 'camera_calibration.npz'), 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs)
    
    return camera_matrix, dist_coeffs

def load_camera_calibration(user_filename:str):
    """Load previously saved calibration data."""
    try:
        data = np.load(os.path.join(CALIBRATION_IMAGE_DIR, user_filename, 'camera_calibration.npz'))
        return data['camera_matrix'], data['dist_coeffs']
    except FileNotFoundError:
        print("No calibration file found. Run calibrate_camera() first.")
        return None, None
    
def take_calibration_photos(user_filename:str, n_photos:int):
    """
    Interactive photo capture with live calibration and quality checking.
    """
    cap = cv.VideoCapture(1)

    # Create output directory
    import os
    os.makedirs(os.path.join(CALIBRATION_DIR, "calibration_images", user_filename), exist_ok=True)

    # Calibration setup
    CHECKERBOARD = (9, 6)
    square_size_mm = 20  # Assuming 20mm squares
    
    # Prepare object points (3D points in real world space)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * (square_size_mm / 1000.0)  # Convert to meters
    
    # Arrays to store calibration data
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    photo_count = 0
    target_photos = n_photos
    current_error = None
    camera_matrix = None
    dist_coeffs = None

    print("Live Calibration Photo Capture")
    print("=============================")
    print("Tips:")
    print("- Keep pattern FLAT (tape to clipboard/book)")
    print("- Fill frame but show all corners")  
    print("- Take from different angles and distances")
    print("- Good lighting, avoid shadows")
    print("- Press SPACE to capture, ESC to finish")
    print("- Calibration will update live as you take photos!")

    while photo_count < target_photos:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Keep original frame for saving (no drawings)
        original_frame = frame.copy()
        
        # Try to find chessboard in real-time
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret_chess, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret_chess:
            # Refine corners for better accuracy
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw corners on display frame (not saved frame)
            cv.drawChessboardCorners(frame, CHECKERBOARD, corners_refined, ret_chess)
            status = f"PATTERN DETECTED - Good to capture! ({photo_count}/{target_photos})"
            color = (0, 255, 0)  # Green
        else:
            status = f"No pattern detected ({photo_count}/{target_photos})"
            color = (0, 0, 255)  # Red
        
        # Display calibration status
        if photo_count >= 4 and current_error is not None:
            error_text = f"Current error: {current_error:.3f} pixels"
            if current_error < 0.5:
                error_color = (0, 255, 0)  # Green - excellent
            elif current_error < 1.0:
                error_color = (0, 255, 255)  # Yellow - good
            else:
                error_color = (0, 0, 255)  # Red - needs improvement
            
            cv.putText(frame, error_text, (10, 70), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, error_color, 2)
        
        # Display status
        cv.putText(frame, status, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv.putText(frame, "SPACE: Capture  ESC: Finish  C: Show current calibration", 
                  (10, frame.shape[0]-20), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv.imshow('Live Calibration Capture', frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord(' ') and ret_chess:  # Space key and pattern detected
            # Save original frame (without drawn corners)
            filename = f'calibration_images/calib_{photo_count:03d}.jpg'
            cv.imwrite(filename, original_frame)
            
            # Add to calibration data
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            
            print(f"✓ Captured {filename}")
            photo_count += 1
            
            # Perform live calibration if we have enough points
            if len(objpoints) >= 4:
                try:
                    ret_cal, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
                        objpoints, imgpoints, gray.shape[::-1], None, None
                    )
                    current_error = ret_cal
                    
                    print(f"  Live calibration: {len(objpoints)} images, error: {current_error:.3f} pixels")
                    
                    # Save intermediate calibration
                    np.savez(f'camera_calibration_live.npz', 
                             camera_matrix=camera_matrix, 
                             dist_coeffs=dist_coeffs,
                             reprojection_error=current_error,
                             num_images=len(objpoints))
                    
                except Exception as e:
                    print(f"  Calibration failed: {e}")
                    current_error = None
            
        elif key == ord('c') and camera_matrix is not None:
            # Show current calibration results
            print(f"\nCurrent Calibration Results ({len(objpoints)} images):")
            print(f"Reprojection Error: {current_error:.4f} pixels")
            print(f"Camera Matrix:\n{camera_matrix}")
            print(f"Distortion Coefficients: {dist_coeffs.flatten()}")
            
        elif key == 27:  # ESC key
            break

    cap.release()
    cv.destroyAllWindows()

    print(f"\nCaptured {photo_count} photos")
    
    # Final calibration
    if len(objpoints) >= 10:
        print("Running final calibration...")
        try:
            ret_final, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            
            print(f"\n" + "="*50)
            print("FINAL CALIBRATION RESULTS")
            print("="*50)
            print(f"Images used: {len(objpoints)}")
            print(f"Reprojection Error: {ret_final:.4f} pixels")
            
            if ret_final < 0.5:
                print("✓ EXCELLENT calibration (error < 0.5)")
            elif ret_final < 1.0:
                print("✓ Good calibration (error < 1.0)")  
            else:
                print("⚠ Acceptable calibration - consider retaking some photos")
            
            print(f"\nCamera Matrix:\n{camera_matrix}")
            print(f"Distortion Coefficients:\n{dist_coeffs.flatten()}")
            
            # Save final calibration
            np.savez(os.path.join(CALIBRATION_DIR, f'camera_calibration_{user_filename}.npz'), 
                     camera_matrix=camera_matrix, 
                     dist_coeffs=dist_coeffs,
                     reprojection_error=ret_final,
                     square_size_mm=square_size_mm,
                     num_images=len(objpoints))
            
            if os.path.isfile(os.path.join(CALIBRATION_DIR, f'camera_calibration_live.npz')):
                os.remove(os.path.join(CALIBRATION_DIR, f'camera_calibration_live.npz'))
                print("Removed temp live calibration file\n")

            print(f"\n✓ Final calibration saved as 'camera_calibration_{user_filename}.npz'")
            return camera_matrix, dist_coeffs
            
        except Exception as e:
            print(f"Final calibration failed: {e}")
            
    else:
        print(f"⚠ Only {len(objpoints)} good images. Need at least 10 for reliable calibration")
        
    return None, None

if __name__ == "__main__":
    print("Camera Calibration System")
    print("========================")
    print("1. Take new calibration photos (with live calibration)")
    print("2. Calibrate from existing photos")
    print("3. Load existing calibration")
    
    choice = input("Choose option (1-3): ").strip()
    filename = input("Type filename for calibration: ")

    if choice == "1":
        n_photos = int(input("# of photos to take: "))
        camera_matrix, dist_coeffs = take_calibration_photos(user_filename=filename, n_photos=n_photos)
        if camera_matrix is not None:
            print("✓ Live calibration completed successfully!")
        
    elif choice == "2":
        camera_matrix, dist_coeffs = calibrate_camera(user_filename=filename)
        if camera_matrix is not None:
            print("✓ Calibration from existing photos completed!")
            
    elif choice == "3":
        camera_matrix, dist_coeffs = load_camera_calibration(user_filename=filename)
        if camera_matrix is not None:
            data = np.load(os.path.join(CALIBRATION_IMAGE_DIR, filename, 'camera_calibration.npz'))
            print(f"✓ Loaded calibration:")
            print(f"  Reprojection Error: {data['reprojection_error']:.4f} pixels")
            print(f"  Images used: {data['num_images']}")
            print(f"✓ Ready to use for ArUco pose estimation!")
            
    else:
        print("Invalid choice! Please start over.")
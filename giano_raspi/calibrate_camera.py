import cv2 as cv
import numpy as np
import glob

PATTERNS = {"chessboard_9x6": "https://github.com/opencv/opencv/blob/master/doc/pattern.png"}
 

def calibrate_camera():
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
    images = glob.glob('calibration_images/*.jpg')  # Put your calibration photos here
    
    if not images:
        print("No calibration images found! Take 15-20 photos of a chessboard pattern.")
        return None, None
    
    for fname in images:
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
    np.savez('camera_calibration.npz', 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs)
    
    return camera_matrix, dist_coeffs

def load_camera_calibration():
    """Load previously saved calibration data."""
    try:
        data = np.load('camera_calibration.npz')
        return data['camera_matrix'], data['dist_coeffs']
    except FileNotFoundError:
        print("No calibration file found. Run calibrate_camera() first.")
        return None, None
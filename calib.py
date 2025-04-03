import cv2
import numpy as np
from picamera2 import Picamera2
import time
from libcamera import Transform

# Calibration settings
CHECKERBOARD = (9, 6)  # number of inner corners per row/column
SQUARE_SIZE = 25.0  # in mm
NUM_IMAGES = 15
WINDOW_SIZE = (1280, 960)

# Termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points like (0,0,0), (1,0,0), ... scaled by square size
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Start camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": WINDOW_SIZE}))
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": WINDOW_SIZE},transform=Transform(hflip=1)))

picam2.start()
time.sleep(2)

cv2.startWindowThread()

count = 0
print("Press SPACE to capture an image for calibration")
print("Press ESC to finish and compute calibration")

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    display = frame.copy()
    if found:
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)

    cv2.putText(display, f"Captured: {count}/{NUM_IMAGES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Calibration", display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32 and found:  # SPACE
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        count += 1
        print(f"Captured {count} images")

    if count >= NUM_IMAGES:
        print("Enough images captured. Press ESC to calibrate.")
        
cv2.destroyAllWindows()

if count >= 5:
    print("Calibrating...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    
    print("\n alibration Complete:")
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist.ravel())

    # Save calibration
    np.savez("calibration_data.npz", camera_matrix=mtx, dist_coeffs=dist)
    print("\n Saved to calibration_data.npz")
else:
    print("Not enough images captured. Try again.")

import cv2
import numpy as np
from picamera2 import Picamera2
import time

# Load camera calibration
with np.load("calibration_data.npz") as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# Marker settings
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
MARKER_LENGTH = 50.0  # in mm (adjust to your printed marker size)

# Start camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()
time.sleep(1)

cv2.startWindowThread()

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect ArUco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    if ids is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH, camera_matrix, dist_coeffs
        )

        for i in range(len(ids)):
            # Draw 3D axes
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], MARKER_LENGTH * 0.5)

            # Get XYZ position
            x, y, z = tvecs[i][0]
            cv2.putText(frame, f"ID:{ids[i][0]} X:{x:.1f} Y:{y:.1f} Z:{z:.1f} mm",
                        (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

    cv2.imshow("ArUco Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break

cv2.destroyAllWindows()

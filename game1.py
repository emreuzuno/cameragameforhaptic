import cv2
import numpy as np
import random
import time
from picamera2 import Picamera2

# === Adjustable Settings ===
WINDOW_SIZE = (640, 480)
MARKER_LENGTH = 50.0  # mm
TARGET_RADIUS = 30  # pixels
REGION_BOUNDS = {
    "x_min": 100, "x_max": 540,
    "y_min": 100, "y_max": 380
}
NUM_TRIALS = 10
TARGET_HOLD_TIME = 1.0  # seconds

# === ArUco Setup ===
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()

# === Load calibration ===
with np.load("calibration_data.npz") as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# === Start Camera ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": WINDOW_SIZE}))
picam2.start()
time.sleep(1)

# === Utility ===
def get_new_target():
    x = random.randint(REGION_BOUNDS["x_min"], REGION_BOUNDS["x_max"])
    y = random.randint(REGION_BOUNDS["y_min"], REGION_BOUNDS["y_max"])
    return (x, y)

def point_inside_circle(px, py, cx, cy, r):
    return (px - cx)**2 + (py - cy)**2 <= r**2

# === Trial Loop ===
cv2.startWindowThread()
trial = 0
target_pos = get_new_target()
inside_since = None

print("🎮 Move the marker to the red circle. Hold for 1 second to score!")

while trial < NUM_TRIALS:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    marker_center = None

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            cv2.aruco.drawDetectedMarkers(frame, corners)
            # Get center point of marker in image coordinates
            corner_pts = corners[i][0]
            cx = int(np.mean(corner_pts[:, 0]))
            cy = int(np.mean(corner_pts[:, 1]))
            marker_center = (cx, cy)
            cv2.circle(frame, marker_center, 5, (255, 0, 0), -1)
            break  # Only first detected marker

    # Draw target
    color = (0, 0, 255)  # Red by default
    if marker_center:
        if point_inside_circle(marker_center[0], marker_center[1], target_pos[0], target_pos[1], TARGET_RADIUS):
            if inside_since is None:
                inside_since = time.time()
            elif time.time() - inside_since >= TARGET_HOLD_TIME:
                # Target hit!
                print(f"✅ Target {trial + 1} reached!")
                trial += 1
                target_pos = get_new_target()
                inside_since = None
                continue  # Skip drawing this frame to move on
            color = (0, 255, 0)  # Green if held
        else:
            inside_since = None

    # Draw target circle
    cv2.circle(frame, target_pos, TARGET_RADIUS, color, -1)

    # Show trial count
    cv2.putText(frame, f"Target: {trial + 1}/{NUM_TRIALS}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 50), 2)

    # Show marker coordinates (optional)
    if marker_center:
        cv2.putText(frame, f"X:{marker_center[0]} Y:{marker_center[1]}",
                    (10, WINDOW_SIZE[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

    cv2.imshow("Haptic Target Challenge", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

print("🎉 Done! All targets completed.")
cv2.destroyAllWindows()

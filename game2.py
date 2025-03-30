import cv2
import numpy as np
import random
import time
import csv
from datetime import datetime
from math import degrees, atan2
from picamera2 import Picamera2

# === Adjustable Settings ===
WINDOW_SIZE = (1280, 960)
MARKER_LENGTH = 50.0  # mm
TARGET_RADIUS = 30  # pixels
REGION_BOUNDS = {
    "x_min": 200, "x_max": 1080,
    "y_min": 200, "y_max": 760
}
NUM_TRIALS = 10
TARGET_HOLD_TIME = 1.0  # seconds

# === Calibration Load ===
with np.load("calibration_data.npz") as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# === ArUco Setup ===
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()

# === Pose Conversion ===
def rvec_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = atan2(R[2, 1], R[2, 2])
        y = atan2(-R[2, 0], sy)
        z = atan2(R[1, 0], R[0, 0])
    else:
        x = atan2(-R[1, 2], R[1, 1])
        y = atan2(-R[2, 0], sy)
        z = 0
    return degrees(z), degrees(y), degrees(x)  # yaw, pitch, roll

# === Utility ===
def get_new_target():
    x = random.randint(REGION_BOUNDS["x_min"], REGION_BOUNDS["x_max"])
    y = random.randint(REGION_BOUNDS["y_min"], REGION_BOUNDS["y_max"])
    return (x, y)

def point_inside_circle(px, py, cx, cy, r):
    return (px - cx)**2 + (py - cy)**2 <= r**2

# === Camera Init ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": WINDOW_SIZE}))
picam2.start()
time.sleep(1)
cv2.startWindowThread()

# === Game State ===
trial_started = False
trial = 0
target_pos = get_new_target()
inside_since = None
start_time = None
session_data = []

print("🔄 Press SPACE to start the session. ESC to exit.")

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    marker_center = None
    x = y = z = yaw = pitch = roll = None

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            cv2.aruco.drawDetectedMarkers(frame, corners)
            corner_pts = corners[i][0]
            cx = int(np.mean(corner_pts[:, 0]))
            cy = int(np.mean(corner_pts[:, 1]))
            marker_center = (cx, cy)

            x, y, z = tvecs[i][0]
            yaw, pitch, roll = rvec_to_euler(rvecs[i])
            break

    if trial_started and trial < NUM_TRIALS:
        color = (0, 0, 255)  # Red by default

        if marker_center:
            if point_inside_circle(marker_center[0], marker_center[1], target_pos[0], target_pos[1], TARGET_RADIUS):
                if inside_since is None:
                    inside_since = time.time()
                    print(f"Hold on second!")
                elif time.time() - inside_since >= TARGET_HOLD_TIME:
                    print(f"✅ Target {trial + 1} reached!")
                    trial += 1
                    target_pos = get_new_target()
                    inside_since = None
                    continue
                color = (0, 255, 0)
            else:
                inside_since = None

            # Log tracking data
            if x is not None:
                t = time.time() - start_time
                session_data.append([
                    f"{t:.3f}", trial + 1,
                    f"{x:.2f}", f"{y:.2f}", f"{z:.2f}",
                    f"{yaw:.2f}", f"{pitch:.2f}", f"{roll:.2f}"
                ])

        # Draw the target circle (no fill)
        cv2.circle(frame, target_pos, TARGET_RADIUS, color, 2)

    else:
        cv2.putText(frame, "Press SPACE to start", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Show trial count
    if trial_started:
        cv2.putText(frame, f"Target: {trial + 1}/{NUM_TRIALS}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 50), 2)

    # Show marker feedback
    if marker_center:
        cv2.circle(frame, marker_center, 4, (255, 0, 0), -1)

    cv2.imshow("Haptic Challenge", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32 and not trial_started:  # SPACE
        print("▶️ Session started.")
        trial_started = True
        trial = 0
        session_data.clear()
        start_time = time.time()
        target_pos = get_new_target()
        inside_since = None

# === Save session data ===
if session_data:
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"haptic_session_{timestamp_str}.csv"
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "Target", "X (mm)", "Y (mm)", "Z (mm)", "Yaw", "Pitch", "Roll"])
        writer.writerows(session_data)
    print(f"💾 Session data saved to {filename}")

cv2.destroyAllWindows()

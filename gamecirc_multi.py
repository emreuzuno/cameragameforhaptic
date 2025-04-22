import cv2
import numpy as np
import random
import time
from picamera2 import Picamera2
import csv
from datetime import datetime
from math import degrees, atan2
from libcamera import Transform

WINDOW_SIZE = (1280, 960)
MARKER_LENGTH = 50.0
REGION_BOUNDS = {
    "x_min": 100, "x_max": 980,
    "y_min": 200, "y_max": 760
}
RADIUS_BOUNDS = {
    "r_min": 67, "r_max": 110
}
NUM_TRIALS = 10
TARGET_HOLD_TIME = 1.0
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()

with np.load("calibration_data.npz") as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": WINDOW_SIZE}))
picam2.start()
time.sleep(1)

def rvec_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = (R[0, 0]**2 + R[1, 0]**2)**0.5
    singular = sy < 1e-6
    if not singular:
        x = atan2(R[2, 1], R[2, 2])
        y = atan2(-R[2, 0], sy)
        z = atan2(R[1, 0], R[0, 0])
    else:
        x = atan2(-R[1, 2], R[1, 1])
        y = atan2(-R[2, 0], sy)
        z = 0
    return degrees(z), degrees(y), degrees(x)

def get_new_target():
    x = random.randint(REGION_BOUNDS["x_min"], REGION_BOUNDS["x_max"])
    y = random.randint(REGION_BOUNDS["y_min"], REGION_BOUNDS["y_max"])
    return (x, y)

def point_inside_circle(px, py, cx, cy, cz, r):
    return (px - cx)**2 + (py - cy)**2 <= r**2 and cz <= r+10 and cz >= r-10

def get_new_radius():
    return random.randint(RADIUS_BOUNDS["r_min"], RADIUS_BOUNDS["r_max"])

cv2.startWindowThread()
trial_started = False
trial = 0
target_pos = get_new_target()
target_r = get_new_radius()
inside_since = None
start_time = None
end_time = None
marker_id = None
recording = False
csv_file = None
csv_writer = None
recording_count = 0
message = "Press SPACE to start session"
message_time = time.time()
MESSAGE_DURATION = 5.0

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    marker_center = None

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            cv2.aruco.drawDetectedMarkers(frame, corners)
            x, y, z = tvecs[i][0]
            rvec = rvecs[i]
            if ids[i][0] == 0:
                cx = int(np.mean(corners[i][0][:, 0]))
                cy = int(np.mean(corners[i][0][:, 1]))
                cz = int(max(40, 200 - z / 10))
                marker_center = (cx, cy)
                cv2.circle(frame, marker_center, cz, (255, 0, 0), -1)
                if recording and csv_writer:
                    timestamp = time.time() - start_time
                    yaw, pitch, roll = rvec_to_euler(rvec)
                    csv_writer.writerow([
                        f"{timestamp:.3f}", marker_id,
                        f"{cx:.2f}", f"{cy:.2f}", f"{cz:.2f}",
                        f"{yaw:.2f}", f"{pitch:.2f}", f"{roll:.2f}",
                        f"{trial:.2f}", f"{target_pos[0]:.2f}", f"{target_pos[1]:.2f}", f"{target_r:.2f}"
                    ])

    if trial_started and trial == NUM_TRIALS:
        if end_time is None:
            end_time = time.time()
            message = f"Task Completed! Score: {end_time - start_time:.2f} sec"
            message_time = time.time()

    if trial_started and trial < NUM_TRIALS:
        color = (0, 0, 255)
        if marker_center:
            if point_inside_circle(marker_center[0], marker_center[1], target_pos[0], target_pos[1], cz, target_r):
                if inside_since is None:
                    inside_since = time.time()
                    message_time = time.time()
                elif time.time() - inside_since >= TARGET_HOLD_TIME:
                    message = f"Target {trial + 1} reached!"
                    message_time = time.time()
                    trial += 1
                    target_pos = get_new_target()
                    target_r = get_new_radius()
                    inside_since = None
                    continue
                color = (0, 255, 0)
            else:
                inside_since = None
        cv2.circle(frame, target_pos, target_r, color, 3)

    frame = cv2.flip(frame, 1)

    if trial_started and trial < NUM_TRIALS:
        cv2.putText(frame, f"Target: {trial}/{NUM_TRIALS}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 50), 2)

    if marker_center:
        cv2.putText(frame, f"ID:{marker_id} X:{x:.1f} Y:{y:.1f} Z:{z:.1f}",
                    (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    status_text = "Recording" if recording else "Not Recording"
    color = (0, 0, 255) if recording else (200, 200, 200)
    cv2.putText(frame, status_text, (100, frame.shape[0] - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if time.time() - message_time <= MESSAGE_DURATION:
        if end_time is not None:
            cv2.putText(frame, message, (200, 450),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, message, (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Target Challenge", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32 and (not trial_started or (trial == NUM_TRIALS and end_time is not None)):
        trial_started = True
        trial = 0
        end_time = None
        target_pos = get_new_target()
        target_r = get_new_radius()
        inside_since = None
        start_time = time.time()
        message = "Session started"
        message_time = time.time()
        if not recording:
            recording_count += 1
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"circlegame_rec{recording_count}_{timestamp_str}.csv"
            csv_file = open(filename, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "Time (s)", "Marker ID",
                "X (mm)", "Y (mm)", "Z (mm)",
                "Yaw (deg)", "Pitch (deg)", "Roll (deg)",
                "Marker No", "X ", "Y ", "R"
            ])
            recording = True
            print(f"Recording started: {filename}")

cv2.destroyAllWindows()

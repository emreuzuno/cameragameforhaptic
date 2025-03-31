import cv2
import numpy as np
import random
import time
from picamera2 import Picamera2
import csv
from datetime import datetime
from math import degrees, atan2, asin
from libcamera import Transform

WINDOW_SIZE = (1280, 960)
# WINDOW_SIZE = (640, 480)

MARKER_LENGTH = 50.0  
TARGET_RADIUS = 30  
REGION_BOUNDS = {
    "x_min": 200, "x_max": 1080,
    "y_min": 200, "y_max": 760
}
RADIUS_BOUNDS = {
    "r_min": 30, "r_max": 60
}
NUM_TRIALS = 10
TARGET_HOLD_TIME = 1.0
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()

with np.load("calibration_data.npz") as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": WINDOW_SIZE},transform=Transform(hflip=1)))
# picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": WINDOW_SIZE}))

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

    return degrees(z), degrees(y), degrees(x)  # yaw, pitch, roll

def get_new_target():
    x = random.randint(REGION_BOUNDS["x_min"], REGION_BOUNDS["x_max"])
    y = random.randint(REGION_BOUNDS["y_min"], REGION_BOUNDS["y_max"])
    return (x, y)

def point_inside_circle(px, py, cx, cy, r):
    return (px - cx)**2 + (py - cy)**2 <= r**2

def get_new_radius():
    r = random.randint(RADIUS_BOUNDS["r_min"], RADIUS_BOUNDS["r_max"])
    return (r)

cv2.startWindowThread()
trial_started = False
trial = 0
target_pos = get_new_target()
target_r = get_new_radius()
inside_since = None
start_time = None

# recording stuff
recording = False
csv_file = None
csv_writer = None
recording_count = 0
start_time = None

message = "Press SPACE to start session"
message_time = time.time()
MESSAGE_DURATION = 2.0  

print("Press SPACE to start the session. ESC to exit.")

while True:

    frame = picam2.capture_array()
    frame = cv2.flip(frame, 1)  
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    marker_center = None

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            cv2.aruco.drawDetectedMarkers(frame, corners)
            corner_pts = corners[i][0]
            cx = int(np.mean(corner_pts[:, 0]))
            cy = int(np.mean(corner_pts[:, 1]))
            marker_id = ids[i][0]
            x, y, z = tvecs[i][0]
            rvec = rvecs[i]
            marker_center = (cx, cy)
            cv2.circle(frame, marker_center, 5, (255, 0, 0), -1)

            if recording and csv_writer:
                timestamp = time.time() - start_time
                yaw, pitch, roll = rvec_to_euler(rvec)
                csv_writer.writerow([
                    f"{timestamp:.3f}", marker_id,
                    f"{x:.2f}", f"{y:.2f}", f"{z:.2f}",
                    f"{yaw:.2f}", f"{pitch:.2f}", f"{roll:.2f}"
                ])

    status_text = "Recording" if recording else "Not Recording"
    color = (0, 0, 255) if recording else (200, 200, 200)
    cv2.putText(frame, status_text, (100, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


     

    if trial_started and trial < NUM_TRIALS:
        color = (0, 0, 255)  
        if marker_center:
            if point_inside_circle(marker_center[0], marker_center[1], target_pos[0], target_pos[1], target_r):
                if inside_since is None:
                    inside_since = time.time()
                    message = "Hold steady for 1 second..."
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

    else:
        cv2.putText(frame, "Press SPACE to start", (450, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


    if trial_started:
        cv2.putText(frame, f"Target: {trial}/{NUM_TRIALS}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 50), 2)

    if marker_center:
        cv2.putText(frame, f"X:{marker_center[0]} Y:{marker_center[1]}",
                    (10, WINDOW_SIZE[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

    if time.time() - message_time <= MESSAGE_DURATION:
        cv2.putText(frame, message, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Target Challenge", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  
        break
    elif key == 32 and not trial_started:  
        trial_started = True
        trial = 0
        target_pos = get_new_target()
        target_r = get_new_radius()
        inside_since = None
        start_time = time.time()
        message = "Session started"
        message_time = time.time()

        if not recording:
            recording_count += 1
            start_time = time.time()
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tracking_record_{recording_count}_{timestamp_str}.csv"
            csv_file = open(filename, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "Time (s)", "Marker ID",
                "X (mm)", "Y (mm)", "Z (mm)",
                "Yaw (deg)", "Pitch (deg)", "Roll (deg)"
            ])
            recording = True
            print(f"Recording started: {filename}")
        else:
            recording = False
            csv_file.close()
            csv_writer = None
            print(f"Recording saved.")

print("Done! All targets completed.")

cv2.destroyAllWindows()

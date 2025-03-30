import cv2
import numpy as np
from picamera2 import Picamera2
import time
import csv
from datetime import datetime
from math import degrees, atan2, asin

# === Helper: Convert rvec to Euler angles ===
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

# === Load Calibration ===
with np.load("calibration_data.npz") as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# === ArUco Setup ===
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
MARKER_LENGTH = 50.0  # mm (adjust to your marker size)

# === Camera Setup ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()
time.sleep(1)

cv2.startWindowThread()
print("🔄 Press SPACE to start/stop recording, ESC to quit.")

# === Recording Control ===
recording = False
csv_file = None
csv_writer = None
recording_count = 0
start_time = None

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                marker_id = ids[i][0]
                x, y, z = tvecs[i][0]
                rvec = rvecs[i]

                # Draw 3D axis
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvecs[i], MARKER_LENGTH * 0.5)

                # Display pose
                cv2.putText(frame, f"ID:{marker_id} X:{x:.1f} Y:{y:.1f} Z:{z:.1f}",
                            (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if recording and csv_writer:
                    timestamp = time.time() - start_time
                    yaw, pitch, roll = rvec_to_euler(rvec)
                    csv_writer.writerow([
                        f"{timestamp:.3f}", marker_id,
                        f"{x:.2f}", f"{y:.2f}", f"{z:.2f}",
                        f"{yaw:.2f}", f"{pitch:.2f}", f"{roll:.2f}"
                    ])

        # Show recording status
        status_text = "● Recording" if recording else "○ Not Recording"
        color = (0, 0, 255) if recording else (200, 200, 200)
        cv2.putText(frame, status_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show frame
        cv2.imshow("ArUco Pose Tracker", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to quit
            break
        elif key == 32:  # SPACE to start/stop recording
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
                print(f"▶️ Recording started: {filename}")
            else:
                recording = False
                csv_file.close()
                csv_writer = None
                print(f"💾 Recording saved.")

finally:
    cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()

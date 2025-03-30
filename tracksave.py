import cv2
import numpy as np
from picamera2 import Picamera2
import time
import csv
from datetime import datetime

# Load camera calibration
with np.load("calibration_data.npz") as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# ArUco setup
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
MARKER_LENGTH = 50.0  # mm

# Camera setup
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()
time.sleep(1)

cv2.startWindowThread()
print("🔄 Press SPACE to start/stop recording, ESC to quit.")

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

                # Draw axis on marker
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], MARKER_LENGTH * 0.5)

                # Show coordinates
                cv2.putText(frame, f"ID:{marker_id} X:{x:.1f} Y:{y:.1f} Z:{z:.1f}",
                            (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Save to CSV if recording
                if recording and csv_writer:
                    timestamp = time.time() - start_time
                    csv_writer.writerow([f"{timestamp:.3f}", marker_id, f"{x:.2f}", f"{y:.2f}", f"{z:.2f}"])

        # Display recording status
        status_text = "● Recording" if recording else "○ Not Recording"
        color = (0, 0, 255) if recording else (200, 200, 200)
        cv2.putText(frame, status_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("ArUco Tracker", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            if not recording:
                # Start new recording
                recording_count += 1
                start_time = time.time()
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tracking_record_{recording_count}_{timestamp_str}.csv"
                csv_file = open(filename, mode='w', newline='')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["Time (s)", "Marker ID", "X (mm)", "Y (mm)", "Z (mm)"])
                recording = True
                print(f"▶️ Recording started: {filename}")
            else:
                # Stop and save
                recording = False
                csv_file.close()
                csv_writer = None
                print(f"💾 Recording saved.")
finally:
    cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()

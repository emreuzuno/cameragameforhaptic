import cv2
import numpy as np
import time
from picamera2 import Picamera2
from libcamera import Transform
from datetime import datetime
from math import degrees, atan2
import cv2.aruco as aruco

# Camera and marker setup
WINDOW_SIZE = (1280, 960)
MARKER_LENGTH = 50.0  # mm

# Ball size bounds
BALL_RADIUS_MIN = 10
BALL_RADIUS_MAX = 30
Z_MIN = 150  # mm
Z_MAX = 400  # mm

# Maze setup
MAZE_WIDTH, MAZE_HEIGHT = 640, 480
START_POS = (50, 50)
GOAL_REGION = ((580, 420), (630, 470))  # Top-left and bottom-right of goal box

# Load calibration
with np.load("calibration_data.npz") as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# Create a simple maze image
maze = np.ones((MAZE_HEIGHT, MAZE_WIDTH), dtype=np.uint8) * 255
cv2.rectangle(maze, (0, 0), (MAZE_WIDTH-1, MAZE_HEIGHT-1), 0, 10)  # border
cv2.rectangle(maze, (100, 100), (540, 120), 0, -1)
cv2.rectangle(maze, (100, 200), (540, 220), 0, -1)
cv2.rectangle(maze, (100, 300), (540, 320), 0, -1)
cv2.rectangle(maze, GOAL_REGION[0], GOAL_REGION[1], 128, -1)  # Goal box (gray)

# Camera setup
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": WINDOW_SIZE}, transform=Transform(hflip=1)))
picam2.start()
time.sleep(1)

# ArUco setup
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)
ARUCO_PARAMS = aruco.DetectorParameters_create()

def map_z_to_radius(z):
    z = np.clip(z, Z_MIN, Z_MAX)
    return int(np.interp(z, [Z_MIN, Z_MAX], [BALL_RADIUS_MAX, BALL_RADIUS_MIN]))

def is_collision(maze_img, cx, cy, r):
    roi = maze_img[max(0, cy - r):min(cy + r, MAZE_HEIGHT), max(0, cx - r):min(cx + r, MAZE_WIDTH)]
    if roi.size == 0:
        return True
    return np.any(roi == 0)

def is_in_goal(cx, cy):
    (x1, y1), (x2, y2) = GOAL_REGION
    return x1 <= cx <= x2 and y1 <= cy <= y2

# Game state
position = START_POS
ball_radius = 20
game_started = False
message = "Press SPACE to start"
message_time = time.time()

cv2.startWindowThread()

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_RGB2GRAY)

    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    if game_started and ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            x, y, z = tvecs[i][0]
            cx = int(np.interp(x, [-100, 100], [0, MAZE_WIDTH]))
            cy = int(np.interp(-y, [-100, 100], [0, MAZE_HEIGHT]))
            radius = map_z_to_radius(z)

            if is_collision(maze, cx, cy, radius):
                position = START_POS
                message = "Hit wall! Resetting..."
                message_time = time.time()
            else:
                position = (cx, cy)
                ball_radius = radius

            if is_in_goal(cx, cy):
                message = "Maze completed!"
                message_time = time.time()
                game_started = False
            break

    # Render
    maze_color = cv2.cvtColor(maze, cv2.COLOR_GRAY2BGR)
    cv2.circle(maze_color, position, ball_radius, (0, 0, 255), -1)
    cv2.rectangle(maze_color, GOAL_REGION[0], GOAL_REGION[1], (0, 255, 0), 2)

    # Message overlay
    if time.time() - message_time < 2:
        cv2.putText(maze_color, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    maze_color = cv2.flip(maze_color, 1) 

    cv2.imshow("3D Maze Game", maze_color)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        game_started = True
        position = START_POS
        message = "Game started!"
        message_time = time.time()

cv2.destroyAllWindows()

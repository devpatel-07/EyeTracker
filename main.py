import sys
import cv2
import numpy as np
import threading
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

import gl_sphere as gl
import PupilDetector as cv_pupil

VIDEO_PATH = "eye_test_custom.mp4"
SCREEN_W = 640
SCREEN_H = 480

# Shared state between CV thread and GL thread
latest_rotated_rect = ((0, 0), (0, 0), 0)
latest_cv_frame = None
lock = threading.Lock()
cv_thread_running = True

import queue
frame_queue = queue.Queue(maxsize=1)

video_finished = False

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('C:/Users/devpa/Downloads/EyeTracker/output.avi', fourcc, 30.0, (SCREEN_W, SCREEN_H))
if not out.isOpened():
    print("Error: VideoWriter failed to open.")
    sys.exit(1)

def overlay_sphere_on_frame(cv_frame, gl_frame):
    gl_bgr = cv2.cvtColor(gl_frame, cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(gl_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    np.copyto(cv_frame, gl_bgr, where=mask[:, :, np.newaxis] > 0)
    return cv_frame

def frame_reader_thread():
    while cv_thread_running:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame_queue.put(frame, block=True)

def cv_thread_func():
    global latest_rotated_rect, latest_cv_frame, cv_thread_running

    while cv_thread_running:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if frame is None:
            print("Video ended.")
            global video_finished
            cv_thread_running = False
            video_finished = True
            break

        rotated_rect = cv_pupil.process_frame(frame)
        cv_frame = cv_pupil.crop_to_aspect_ratio(frame)

        with lock:
            latest_rotated_rect = rotated_rect
            latest_cv_frame = cv_frame

def tick():
    
    global video_finished
    if video_finished:
        shutdown()
        return

    # Grab latest CV results without blocking
    with lock:
        rotated_rect = latest_rotated_rect
        cv_frame = latest_cv_frame

    if cv_frame is None:
        return

    if rotated_rect == ((0, 0), (0, 0), 0):
        cv2.imshow("Eye Tracker", cv_frame)
        cv2.waitKey(1)
        return

    (pupil_x, pupil_y), axes, angle = rotated_rect

    # Get eye center estimate from PupilDetector
    if len(cv_pupil.eye_centers) > 0:
        center_x = sum([p[0] for p in cv_pupil.eye_centers]) // len(cv_pupil.eye_centers)
        center_y = sum([p[1] for p in cv_pupil.eye_centers]) // len(cv_pupil.eye_centers)
    else:
        center_x = SCREEN_W // 2
        center_y = SCREEN_H // 2

    # Update GL sphere rotation
    gl_frame = gl.update_sphere_rotation(
        x=int(pupil_x),
        y=int(pupil_y),
        center_x=center_x,
        center_y=center_y,
        screen_width=SCREEN_W,
        screen_height=SCREEN_H
    )

    if gl_frame is None:
        cv2.waitKey(1)
        return

    # Resize GL frame if needed
    if gl_frame.shape[:2] != cv_frame.shape[:2]:
        gl_frame = cv2.resize(gl_frame, (cv_frame.shape[1], cv_frame.shape[0]))

    # Draw pupil ellipse on CV frame
    cv_frame = cv_frame.copy()  # don't modify the shared frame
    ellipse = (
        (float(pupil_x), float(pupil_y)),
        (float(axes[0]), float(axes[1])),
        float(angle)
    )
    cv2.ellipse(cv_frame, ellipse, (55, 255, 0), 2)

    # Composite and display
    combined = overlay_sphere_on_frame(cv_frame, gl_frame)
    cv2.imshow("Eye Tracker", combined)
    out.write(combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        shutdown()
    elif key == ord(' '):
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '):
                break
            elif k == ord('q'):
                shutdown()
                return

def shutdown():
    global cv_thread_running
    cv_thread_running = False
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    QApplication.quit()

if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: could not open video at {VIDEO_PATH}")
        sys.exit(1)

    # Start CV pipeline on its own thread
    reader = threading.Thread(target=frame_reader_thread, daemon=True)
    reader.start()

    t = threading.Thread(target=cv_thread_func, daemon=True)
    t.start()

    # Boot OpenGL window on main thread (Qt requires this)
    app = gl.start_gl_window()

    # QTimer drives GL updates on the main thread
    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(16)  # ~60fps

    sys.exit(app.exec_())
"""
eye_tracker_3d.py
=================
Connects to pupiltracker.py for real pupil detection and eye-centre estimation,
then computes and displays the 3D gaze vector.

Both files must be in the same folder.

Usage
-----
    python eye_tracker_3d.py               # webcam
    python eye_tracker_3d.py eye.mp4       # video file

Controls
--------
    Q      = quit
    SPACE  = pause / resume
"""

import sys
import math
import random
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog

# ── Import pupil tracker ──────────────────────────────────────────────────────
import PupilDetector

try:
    import gl_sphere
    GL_SPHERE_AVAILABLE = True
except ImportError:
    GL_SPHERE_AVAILABLE = False
    print("gl_sphere not found — OpenGL sphere disabled.")


# ══════════════════════════════════════════════════════════════════════════════
#  Gaze vector computation
#  Takes REAL pixel coords from pupiltracker and returns a 3D direction.
# ══════════════════════════════════════════════════════════════════════════════

def compute_gaze_vector(pupil_x, pupil_y, eye_center_x, eye_center_y,
                        screen_width=640, screen_height=480):
    """
    Given the detected pupil pixel (pupil_x, pupil_y) and the estimated
    2-D eye-ball centre (eye_center_x, eye_center_y), return:
        sphere_center   — 3D world-space position of the eye sphere centre
        gaze_direction  — normalised 3D gaze direction vector

    The eye-centre pixel is converted to an OpenGL-style world offset so
    the sphere sits at the right place in 3D, then the pupil pixel casts
    a ray through that sphere to find the gaze direction.
    """
    fov_y_deg    = 45.0
    aspect_ratio = screen_width / screen_height
    far_clip     = 100.0
    camera_pos   = np.array([0.0, 0.0, 3.0])

    fov_y_rad       = np.radians(fov_y_deg)
    half_h_far      = np.tan(fov_y_rad / 2) * far_clip
    half_w_far      = half_h_far * aspect_ratio

    # ── Pupil pixel → NDC → far-plane world point → ray direction ────────────
    ndc_x = (2.0 * pupil_x) / screen_width  - 1.0
    ndc_y = 1.0 - (2.0 * pupil_y) / screen_height

    far_pt        = np.array([ndc_x * half_w_far, ndc_y * half_h_far,
                               camera_pos[2] - far_clip])
    ray_dir       = far_pt - camera_pos
    ray_dir      /= np.linalg.norm(ray_dir)
    # flip so ray points toward the eye (into the scene)
    ray_dir       = -ray_dir

    # ── Eye-centre pixel → world-space sphere centre ──────────────────────────
    cx_ndc = (eye_center_x / screen_width)  * 2.0 - 1.0
    cy_ndc = 1.0 - (eye_center_y / screen_height) * 2.0
    sphere_center = np.array([cx_ndc * 1.5, cy_ndc * 1.5, 0.0])

    # ── Ray-sphere intersection ───────────────────────────────────────────────
    inner_radius = 1.0 / 1.05
    origin    = camera_pos
    direction = -ray_dir          # direction toward scene
    L = origin - sphere_center

    a    = np.dot(direction, direction)
    b    = 2 * np.dot(direction, L)
    c    = np.dot(L, L) - inner_radius ** 2
    disc = b * b - 4 * a * c

    if disc < 0:
        # Ray missed the sphere — use closest point on sphere surface
        t = -np.dot(direction, L) / a
        intersection_pt = origin + t * direction
    else:
        sq = np.sqrt(disc)
        t1 = (-b - sq) / (2 * a)
        t2 = (-b + sq) / (2 * a)
        t  = None
        if t1 > 0 and t2 > 0:
            t = min(t1, t2)
        elif t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        if t is None:
            return None, None
        intersection_pt = origin + t * direction

    # ── Compute gaze direction from intersection ──────────────────────────────
    local     = intersection_pt - sphere_center
    local    /= np.linalg.norm(local)

    # Rotate the local +Z axis to align with the intersection direction
    z_axis = np.array([0.0, 0.0, inner_radius])
    z_axis /= np.linalg.norm(z_axis)

    rot_axis      = np.cross(z_axis, local)
    rot_axis_norm = np.linalg.norm(rot_axis)
    if rot_axis_norm < 1e-6:
        return sphere_center, local

    rot_axis /= rot_axis_norm
    dot       = np.clip(np.dot(z_axis, local), -1.0, 1.0)
    angle     = np.arccos(dot)

    c_ = np.cos(angle);  s_ = np.sin(angle);  t_ = 1 - c_
    rx, ry, rz = rot_axis
    R = np.array([
        [t_*rx*rx + c_,      t_*rx*ry - s_*rz,  t_*rx*rz + s_*ry],
        [t_*rx*ry + s_*rz,   t_*ry*ry + c_,     t_*ry*rz - s_*rx],
        [t_*rx*rz - s_*ry,   t_*ry*rz + s_*rx,  t_*rz*rz + c_   ],
    ])

    gaze = R @ np.array([0.0, 0.0, inner_radius])
    gaze /= np.linalg.norm(gaze)

    return sphere_center, gaze


# ══════════════════════════════════════════════════════════════════════════════
#  Binocular convergence
# ══════════════════════════════════════════════════════════════════════════════

def compute_gaze_intersection(l_center, l_dir, r_center, r_dir):
    """
    Finds the 3D midpoint of closest approach between two gaze rays (skew lines).
    Returns the estimated 3D fixation point.
    """
    delta = l_center - r_center

    dll = np.dot(l_dir, l_dir)
    dlr = np.dot(l_dir, r_dir)
    drr = np.dot(r_dir, r_dir)
    denom = dll * drr - dlr * dlr

    if abs(denom) < 1e-6:
        # Parallel rays — return a far point straight ahead
        mid = (l_center + r_center) / 2
        avg = (l_dir + r_dir) / 2
        avg /= np.linalg.norm(avg)
        return mid + avg * 1000.0

    t_l = (dlr * np.dot(r_dir, delta) - drr * np.dot(l_dir, delta)) / denom
    t_r = (dll * np.dot(r_dir, delta) - dlr * np.dot(l_dir, delta)) / denom

    pt_l = l_center + t_l * l_dir
    pt_r = r_center + t_r * r_dir
    return (pt_l + pt_r) / 2


# ══════════════════════════════════════════════════════════════════════════════
#  HUD drawing
# ══════════════════════════════════════════════════════════════════════════════

def draw_hud(frame, sphere_center, gaze_dir, gaze_point,
             pupil_x, pupil_y, eye_cx, eye_cy):
    """Burn gaze data onto the frame as a semi-transparent HUD."""
    H, W = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H - 105), (W, H), (5, 8, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.line(frame, (0, H - 105), (W, H - 105), (0, 120, 220), 1)

    if sphere_center is not None and gaze_dir is not None:
        lines = [
            (f"Eye ctr 3D : ({sphere_center[0]:+.3f}, {sphere_center[1]:+.3f},"
             f" {sphere_center[2]:+.3f})",                          (0, 255, 180)),
            (f"Gaze dir   : ({gaze_dir[0]:+.4f}, {gaze_dir[1]:+.4f},"
             f" {gaze_dir[2]:+.4f})",                               (255, 220,   0)),
            (f"Pupil 2D   : ({pupil_x:3d}, {pupil_y:3d}) px   "
             f"Eye ctr 2D : ({eye_cx}, {eye_cy}) px",               (160, 200, 255)),
        ]
        if gaze_point is not None:
            lines.append(
                (f"3D target  : ({gaze_point[0]:+.2f}, {gaze_point[1]:+.2f},"
                 f" {gaze_point[2]:+.2f})",                         (0, 180, 255))
            )
        for i, (txt, col) in enumerate(lines):
            y = H - 88 + i * 22
            cv2.putText(frame, txt, (11, y+1), font, 0.42, (0,   0,   0), 2)
            cv2.putText(frame, txt, (10, y),   font, 0.42, col,          1)


# ══════════════════════════════════════════════════════════════════════════════
#  Per-frame processing  —  calls pupiltracker then computes 3D gaze
# ══════════════════════════════════════════════════════════════════════════════

def process_frame_3d(frame):
    """
    1. Run pupiltracker.process_frame() to get the detected ellipse.
    2. Read back the pupil centre and eye-centre estimate from pupiltracker's
       shared state.
    3. Compute the 3D gaze vector and draw everything on the frame.

    Returns (sphere_center, gaze_direction) or (None, None).
    """
    # ── Step 1: pupil detection (modifies frame in-place via its own imshow) ──
    ellipse = PupilDetector.process_frame(frame)

    if ellipse is None:
        return None, None

    # ── Step 2: extract pupil centre from the returned ellipse ───────────────
    (pupil_x, pupil_y), _, _ = ellipse
    pupil_x = int(pupil_x)
    pupil_y = int(pupil_y)

    # ── Step 3: get eye-centre estimate from pupiltracker's running average ───
    # pupiltracker accumulates eye_centers[] each frame; use its average.
    if len(PupilDetector.eye_centers) > 0:
        eye_cx = int(sum(p[0] for p in PupilDetector.eye_centers)
                     / len(PupilDetector.eye_centers))
        eye_cy = int(sum(p[1] for p in PupilDetector.eye_centers)
                     / len(PupilDetector.eye_centers))
    else:
        # Fallback before enough frames have accumulated
        eye_cx, eye_cy = frame.shape[1] // 2, frame.shape[0] // 2

    # ── Step 4: compute 3D gaze ───────────────────────────────────────────────
    sphere_center, gaze_dir = compute_gaze_vector(
        pupil_x, pupil_y, eye_cx, eye_cy,
        frame.shape[1], frame.shape[0])

    if sphere_center is None:
        return None, None

    # ── Step 5: binocular intersection (single-cam: both rays are the same) ──
    # When you add a second camera, pass its pupil/eye-centre here instead.
    gaze_point = compute_gaze_intersection(
        sphere_center, gaze_dir, sphere_center, gaze_dir)

    # ── Step 6: draw on frame ─────────────────────────────────────────────────
    r_px = max(1, int(202))   # eye sphere radius in pixels
    cv2.circle(frame, (eye_cx, eye_cy), r_px, (255, 50, 50), 2)
    cv2.circle(frame, (eye_cx, eye_cy),    8, (255, 255,  0), -1)

    # Eye centre → pupil line
    cv2.line(frame, (eye_cx, eye_cy), (pupil_x, pupil_y), (255, 150, 50), 2)

    # Extended gaze ray arrow
    dx  = pupil_x - eye_cx
    dy  = pupil_y - eye_cy
    ex  = int(eye_cx + 2.2 * dx)
    ey  = int(eye_cy + 2.2 * dy)
    cv2.arrowedLine(frame, (pupil_x, pupil_y), (ex, ey),
                    (80, 255, 80), 3, tipLength=0.22)
    cv2.circle(frame, (pupil_x, pupil_y), 5, (0, 200, 255), -1)

    # Ellipse outline
    (ex2, ey2), (ew, eh), ea = ellipse
    if ew > 0 and eh > 0 and np.isfinite(ex2) and np.isfinite(ey2):
        cv2.ellipse(frame, ellipse, (20, 255, 255), 2)

    # HUD
    draw_hud(frame, sphere_center, gaze_dir, gaze_point,
             pupil_x, pupil_y, eye_cx, eye_cy)

    # OpenGL sphere (if available)
    if GL_SPHERE_AVAILABLE:
        gl_sphere.update_sphere_rotation(
            pupil_x, pupil_y, eye_cx, eye_cy,
            frame.shape[1], frame.shape[0])

    # Terminal log
    print(f"Pupil 2D  : ({pupil_x}, {pupil_y})")
    print(f"Eye ctr 2D: ({eye_cx}, {eye_cy})")
    print(f"Gaze dir  : ({gaze_dir[0]:+.3f}, {gaze_dir[1]:+.3f},"
          f" {gaze_dir[2]:+.3f})")

    # Write gaze_vector.txt  (format: sx,sy,sz,gx,gy,gz)
    try:
        with open("gaze_vector.txt", "w") as f:
            vals = list(sphere_center) + list(gaze_dir)
            f.write(",".join(f"{v:.6f}" for v in vals) + "\n")
    except Exception as e:
        print("Write error:", e)

    cv2.imshow("Eye Tracker 3D — Gaze", frame)

    return sphere_center, gaze_dir


# ══════════════════════════════════════════════════════════════════════════════
#  Camera / video loops
# ══════════════════════════════════════════════════════════════════════════════

selected_camera = None

def process_camera():
    global selected_camera
    cam_index = int(selected_camera.get()) if selected_camera else 0

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    if GL_SPHERE_AVAILABLE:
        gl_sphere.start_gl_window()

    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 0)
            process_frame_3d(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


def process_video_file(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    if GL_SPHERE_AVAILABLE:
        gl_sphere.start_gl_window()

    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            process_frame_3d(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════════════════════

def detect_cameras(max_cams=10):
    available = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def selection_gui():
    global selected_camera
    cameras = detect_cameras()

    root = tk.Tk()
    root.title("Eye Tracker 3D")
    tk.Label(root, text="Orlosky Eye Tracker 3D",
             font=("Arial", 12, "bold")).pack(pady=10)
    tk.Label(root, text="Select Camera:").pack(pady=5)

    selected_camera = tk.StringVar(value="0")
    ttk.Combobox(root, textvariable=selected_camera,
                 values=[str(c) for c in cameras]).pack(pady=5)

    tk.Button(root, text="Start Camera",
              command=lambda: [root.destroy(), process_camera()]).pack(pady=5)

    def browse_video():
        path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4")])
        if path:
            root.destroy()
            process_video_file(path)

    tk.Button(root, text="Browse Video", command=browse_video).pack(pady=5)

    root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        src = sys.argv[1]
        try:
            process_camera_index = int(src)
            selected_camera_val  = str(process_camera_index)

            class _FakeStrVar:
                def get(self): return selected_camera_val
            selected_camera = _FakeStrVar()
            process_camera()
        except ValueError:
            process_video_file(src)
    else:
        selection_gui()
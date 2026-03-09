"""
visualize_3d_gaze.py
====================
Drop-in visualization layer for the existing eye tracker.

Shows:
  - LEFT panel  : CV camera feed with pupil ellipse, eye circle, and gaze arrow
  - RIGHT panel : Live 3D scatter plot of the gaze vector on the unit sphere
                  (matplotlib, updated every frame)
  - BOTTOM-LEFT : Live 3D (x,y,z) coordinate readout burned into the CV frame

Usage
-----
    python visualize_3d_gaze.py                  # uses webcam
    python visualize_3d_gaze.py eye.mp4          # uses video file

The script imports your existing modules (eye_tracker_3d / pupil_detection).
If they are not on the path it falls back to its own pupil detector so you
can test immediately.
"""

import sys
import os
import math
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")          # off-screen so we can convert to a cv2 image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3d projection)

# ── Try to import your existing pipeline ─────────────────────────────────────
try:
    import importlib.util, pathlib

    # Accept both filenames seen in the pasted code
    for candidate in ["eye_tracker_3d.py", "pupil_detection.py"]:
        spec = importlib.util.spec_from_file_location("eye_tracker_3d", candidate)
        if spec:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            process_frame_external = mod.process_frame
            compute_gaze_vector    = mod.compute_gaze_vector
            print(f"[INFO] Loaded pipeline from '{candidate}'")
            break
    else:
        raise ImportError("Neither eye_tracker_3d.py nor pupil_detection.py found")

    USE_EXTERNAL = True

except Exception as e:
    print(f"[WARN] Could not load external pipeline ({e}). Using built-in detector.")
    USE_EXTERNAL = False


# ── Built-in fallback pupil detector ─────────────────────────────────────────

def _builtin_detect_pupil(frame):
    """Returns (ellipse, center_x, center_y) or (None, None, None)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Find darkest region
    min_val, _, min_loc, _ = cv2.minMaxLoc(blurred)
    threshold = int(min_val) + 20
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    # Mask to region around darkest point
    mask = np.zeros_like(binary)
    cx0, cy0 = min_loc
    half = 120
    x1, y1 = max(0, cx0-half), max(0, cy0-half)
    x2, y2 = min(binary.shape[1], cx0+half), min(binary.shape[0], cy0+half)
    mask[y1:y2, x1:x2] = 255
    binary = cv2.bitwise_and(binary, mask)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    # Pick largest contour that isn't too elongated
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ratio = max(w, h) / max(min(w, h), 1)
        if ratio > 3.5:
            continue
        if area > best_area:
            best_area = area
            best = c

    if best is None or len(best) < 5:
        return None, None, None

    ellipse = cv2.fitEllipse(best)
    cx, cy = map(int, ellipse[0])
    return ellipse, cx, cy


# ── Gaze vector computation (mirrors compute_gaze_vector in eye_tracker_3d) ──

def _builtin_compute_gaze(px, py, eye_cx, eye_cy, W=640, H=480):
    """
    Returns (sphere_center_3d, gaze_direction_3d) both as np.ndarray(3,).
    Coordinate convention: OpenGL-style (Y up, Z toward viewer).
    """
    fov_y_deg   = 45.0
    aspect      = W / H
    far_clip    = 100.0
    cam_pos     = np.array([0.0, 0.0, 3.0])

    fov_y_rad   = np.radians(fov_y_deg)
    hh_far      = np.tan(fov_y_rad / 2) * far_clip
    hw_far      = hh_far * aspect

    # NDC of pupil pixel
    ndc_x = (2.0 * px) / W - 1.0
    ndc_y = 1.0 - (2.0 * py) / H

    far_pt    = np.array([ndc_x * hw_far, ndc_y * hh_far, cam_pos[2] - far_clip])
    ray_dir   = far_pt - cam_pos
    ray_dir  /= np.linalg.norm(ray_dir)
    ray_dir   = -ray_dir

    inner_r   = 1.0 / 1.05

    # Eye sphere center in world space
    ox = (eye_cx / W) * 2.0 - 1.0
    oy = 1.0 - (eye_cy / H) * 2.0
    sph_ctr   = np.array([ox * 1.5, oy * 1.5, 0.0])

    origin    = cam_pos
    direction = -ray_dir
    L         = origin - sph_ctr
    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, L)
    c = np.dot(L, L) - inner_r**2
    disc = b*b - 4*a*c

    if disc < 0:
        t = -np.dot(direction, L) / a
    else:
        sq = np.sqrt(disc)
        t1, t2 = (-b - sq)/(2*a), (-b + sq)/(2*a)
        t = min((v for v in (t1, t2) if v > 0), default=None)
        if t is None:
            return None, None

    inter   = origin + t * direction
    local   = inter - sph_ctr
    local  /= np.linalg.norm(local)

    # Compute rotation from +Z to local direction
    z_axis  = np.array([0.0, 0.0, 1.0])
    axis    = np.cross(z_axis, local)
    ax_norm = np.linalg.norm(axis)
    if ax_norm < 1e-6:
        return sph_ctr, local

    axis   /= ax_norm
    angle   = np.arccos(np.clip(np.dot(z_axis, local), -1, 1))
    c_a, s_a, t_a = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    x_, y_, z_ = axis
    R = np.array([
        [t_a*x_*x_ + c_a,       t_a*x_*y_ - s_a*z_,  t_a*x_*z_ + s_a*y_],
        [t_a*x_*y_ + s_a*z_,    t_a*y_*y_ + c_a,      t_a*y_*z_ - s_a*x_],
        [t_a*x_*z_ - s_a*y_,    t_a*y_*z_ + s_a*x_,   t_a*z_*z_ + c_a   ],
    ])
    gaze = R @ np.array([0.0, 0.0, inner_r])
    gaze /= np.linalg.norm(gaze)
    return sph_ctr, gaze


# ── Matplotlib 3D sphere panel ────────────────────────────────────────────────

class GazeSphere3D:
    """Renders a 3D unit sphere + current gaze vector as an off-screen
       matplotlib figure and returns it as a BGR cv2 image."""

    def __init__(self, size=480):
        self.size  = size
        self.fig   = plt.figure(figsize=(size/100, size/100), dpi=100,
                                facecolor="#0a0a0f")
        self.ax    = self.fig.add_subplot(111, projection="3d",
                                          facecolor="#0a0a0f")
        self._draw_sphere_wireframe()
        self.gaze_quiver  = None
        self.gaze_dot     = None
        self.history_pts  = []          # list of (x,y,z) endpoint traces
        self.max_history  = 60

    def _draw_sphere_wireframe(self):
        ax = self.ax
        u = np.linspace(0, 2*np.pi, 24)
        v = np.linspace(0, np.pi, 16)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color="#1a2a4a", linewidth=0.4, alpha=0.5)

        # Equator ring
        theta = np.linspace(0, 2*np.pi, 120)
        ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
                color="#00aaff", linewidth=1.2, alpha=0.7)

        # Axes stubs
        for vec, col, lbl in [
            ([1.3,0,0], "#ff4466", "X"),
            ([0,1.3,0], "#44ff88", "Y"),
            ([0,0,1.3], "#4488ff", "Z"),
        ]:
            ax.quiver(0,0,0, *vec, color=col, linewidth=1.5, arrow_length_ratio=0.15)
            ax.text(*[v*1.45 for v in vec], lbl, color=col,
                    fontsize=8, fontweight="bold", ha="center", va="center")

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_zlim(-1.6, 1.6)
        ax.set_box_aspect([1,1,1])
        ax.axis("off")
        ax.set_title("Gaze Vector — 3D", color="#aaccff",
                     fontsize=10, pad=4, fontfamily="monospace")

    def update(self, sphere_center, gaze_dir):
        ax = self.ax

        # Remove old gaze artists
        if self.gaze_quiver:
            try: self.gaze_quiver.remove()
            except: pass
        if self.gaze_dot:
            try: self.gaze_dot.remove()
            except: pass
        for artist in getattr(self, "_history_artists", []):
            try: artist.remove()
            except: pass
        self._history_artists = []

        gx, gy, gz = float(gaze_dir[0]), float(gaze_dir[1]), float(gaze_dir[2])

        # Gaze arrow
        self.gaze_quiver = ax.quiver(
            0, 0, 0, gx, gy, gz,
            color="#ffdd00", linewidth=2.5,
            arrow_length_ratio=0.18, zorder=10
        )

        # Dot at gaze endpoint
        self.gaze_dot = ax.scatter(
            [gx], [gy], [gz],
            color="#ff8800", s=45, zorder=11, depthshade=False
        )

        # History trace
        self.history_pts.append((gx, gy, gz))
        if len(self.history_pts) > self.max_history:
            self.history_pts.pop(0)

        if len(self.history_pts) > 1:
            hx = [p[0] for p in self.history_pts]
            hy = [p[1] for p in self.history_pts]
            hz = [p[2] for p in self.history_pts]
            n  = len(hx)
            for i in range(1, n):
                alpha = 0.15 + 0.55 * (i / n)
                a, = ax.plot(hx[i-1:i+1], hy[i-1:i+1], hz[i-1:i+1],
                             color="#ff8800", alpha=alpha, linewidth=1.2)
                self._history_artists.append(a)

        self.fig.canvas.draw()

    def to_cv2(self):
        """Return the current figure as a BGR uint8 numpy array."""
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


# ── HUD overlay helpers ───────────────────────────────────────────────────────

def draw_hud(frame, sphere_center, gaze_dir, pupil_xy, eye_center_xy):
    """Burn 3D coordinate info onto the bottom of the CV frame."""
    h, w = frame.shape[:2]

    # Semi-transparent dark bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-100), (w, h), (5, 8, 18), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    mono   = cv2.FONT_HERSHEY_PLAIN

    if sphere_center is not None and gaze_dir is not None:
        sc = sphere_center
        gd = gaze_dir
        texts = [
            (f"Eye ctr  3D : ({sc[0]:+.3f},  {sc[1]:+.3f},  {sc[2]:+.3f})",
             (0, 255, 180)),
            (f"Gaze dir 3D : ({gd[0]:+.3f},  {gd[1]:+.3f},  {gd[2]:+.3f})",
             (255, 220, 0)),
            (f"Pupil 2D : ({pupil_xy[0]}, {pupil_xy[1]})   "
             f"Eye ctr 2D : ({eye_center_xy[0]}, {eye_center_xy[1]})",
             (160, 200, 255)),
        ]
        for i, (txt, col) in enumerate(texts):
            y_pos = h - 78 + i * 26
            # shadow
            cv2.putText(frame, txt, (11, y_pos+1), font, 0.46, (0,0,0), 2)
            cv2.putText(frame, txt, (10, y_pos),   font, 0.46, col,     1)

    # Thin top border line on the bar
    cv2.line(frame, (0, h-100), (w, h-100), (0, 120, 220), 1)


def draw_gaze_on_frame(frame, pupil_x, pupil_y, eye_cx, eye_cy,
                        ellipse=None, max_r=202):
    """Draws eye circle, gaze line and ellipse on the CV frame."""
    # Eye sphere circle
    cv2.circle(frame, (eye_cx, eye_cy), max_r, (255, 50, 50), 2)
    cv2.circle(frame, (eye_cx, eye_cy), 8,     (255, 255, 0), -1)

    if ellipse is not None:
        cv2.ellipse(frame, ellipse, (20, 255, 255), 2)

    if pupil_x is not None and pupil_y is not None:
        # Line: eye center → pupil
        cv2.line(frame, (eye_cx, eye_cy), (pupil_x, pupil_y),
                 (255, 150, 50), 2)
        # Extended gaze ray beyond pupil
        dx = pupil_x - eye_cx
        dy = pupil_y - eye_cy
        ex = int(eye_cx + 2.2 * dx)
        ey = int(eye_cy + 2.2 * dy)
        cv2.arrowedLine(frame, (pupil_x, pupil_y), (ex, ey),
                        (80, 255, 80), 3, tipLength=0.25)
        cv2.circle(frame, (pupil_x, pupil_y), 5, (0, 200, 255), -1)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    # Try to read one frame to get dimensions
    ret, sample = cap.read()
    if not ret:
        print("[ERROR] Cannot read from source.")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    panel_h = 480
    panel_w = 640
    sphere_vis = GazeSphere3D(size=panel_h)

    # Rolling average for eye center
    eye_center_history = []
    MAX_EYE_HIST = 150

    print("\n[Eye Tracker 3D Visualizer]")
    print("  SPACE  = pause/resume")
    print("  Q      = quit\n")

    paused = False
    frame_count = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1

            # Resize to standard panel size
            frame = cv2.resize(frame, (panel_w, panel_h))

            # ── Pupil detection ───────────────────────────────────────────
            ellipse, pupil_x, pupil_y = _builtin_detect_pupil(frame)

            eye_cx, eye_cy = panel_w // 2, panel_h // 2  # fallback

            # ── Eye center estimation (simple moving average of pupil) ────
            if pupil_x is not None:
                eye_center_history.append((pupil_x, pupil_y))
                if len(eye_center_history) > MAX_EYE_HIST:
                    eye_center_history.pop(0)
                if len(eye_center_history) >= 10:
                    eye_cx = int(np.mean([p[0] for p in eye_center_history]))
                    eye_cy = int(np.mean([p[1] for p in eye_center_history]))

            # ── Gaze vector computation ───────────────────────────────────
            sphere_center, gaze_dir = None, None
            if pupil_x is not None:
                if USE_EXTERNAL:
                    try:
                        sphere_center, gaze_dir = compute_gaze_vector(
                            pupil_x, pupil_y, eye_cx, eye_cy,
                            panel_w, panel_h
                        )
                    except Exception as e:
                        print(f"[WARN] External compute_gaze_vector failed: {e}")
                        sphere_center, gaze_dir = _builtin_compute_gaze(
                            pupil_x, pupil_y, eye_cx, eye_cy, panel_w, panel_h
                        )
                else:
                    sphere_center, gaze_dir = _builtin_compute_gaze(
                        pupil_x, pupil_y, eye_cx, eye_cy, panel_w, panel_h
                    )

            # ── Draw CV overlays ──────────────────────────────────────────
            draw_gaze_on_frame(frame, pupil_x, pupil_y, eye_cx, eye_cy,
                               ellipse=ellipse)

            # ── HUD ───────────────────────────────────────────────────────
            pxy = (pupil_x if pupil_x else 0, pupil_y if pupil_y else 0)
            draw_hud(frame, sphere_center, gaze_dir,
                     pxy, (eye_cx, eye_cy))

            # ── Update 3D sphere panel ────────────────────────────────────
            if gaze_dir is not None and frame_count % 2 == 0:   # every 2nd frame
                sphere_vis.update(sphere_center, gaze_dir)
            sphere_img = sphere_vis.to_cv2()
            sphere_img = cv2.resize(sphere_img, (panel_w, panel_h))

            # ── Compose side-by-side ──────────────────────────────────────
            divider = np.full((panel_h, 4, 3), (30, 60, 100), dtype=np.uint8)
            combined = np.hstack([frame, divider, sphere_img])

            # Title bar
            title_bar = np.zeros((36, combined.shape[1], 3), dtype=np.uint8)
            title_bar[:] = (10, 14, 28)
            cv2.putText(title_bar,
                        "EYE TRACKER 3D  |  CV Feed                              "
                        "Gaze Vector Sphere",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (100, 180, 255), 1)
            cv2.line(title_bar, (0, 35), (title_bar.shape[1], 35),
                     (0, 100, 200), 1)

            display = np.vstack([title_bar, combined])

            cv2.imshow("Eye Tracker 3D — Gaze Visualizer", display)

            # Print to terminal every 15 frames
            if frame_count % 15 == 0 and gaze_dir is not None:
                sc = sphere_center
                gd = gaze_dir
                print(f"[{frame_count:05d}]  "
                      f"Pupil2D=({pxy[0]:3d},{pxy[1]:3d})  "
                      f"EyeCtr3D=({sc[0]:+.3f},{sc[1]:+.3f},{sc[2]:+.3f})  "
                      f"Gaze=({gd[0]:+.3f},{gd[1]:+.3f},{gd[2]:+.3f})")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("[PAUSED]" if paused else "[RESUMED]")

    cap.release()
    cv2.destroyAllWindows()
    plt.close("all")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    # Try to parse as integer (webcam index)
    try:
        src = int(src)
    except ValueError:
        pass
    run(src)
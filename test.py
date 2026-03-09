"""
eye_tracker_combined.py
=======================
Clean pipeline — my code only:
  - Simple dark-blob pupil detector (robust, no external dependencies)
  - Physically accurate pixel → mm → ray → sphere intersection (mm-space)
  - Side-by-side CV feed + live matplotlib 3D gaze sphere
  - HUD with real mm-space 3D coordinates

Coordinate system (right-handed):
  - Eye centre C = (0, 0, 0) mm
  - Camera at  O = (0, 0, +camera_z_mm), looking toward -Z
  - X = right,  Y = up,  Z = toward camera

Usage
-----
    python eye_tracker_combined.py              # webcam index 0
    python eye_tracker_combined.py eye.mp4      # video file

Controls
--------
    SPACE  = pause / resume
    Q      = quit
    D      = toggle debug window
"""

import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")          # off-screen rendering — no GUI window for mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3-D projection


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Camera & Eye Parameters  —  edit these to match your hardware
# ══════════════════════════════════════════════════════════════════════════════

class EyeTrackingParams:
    # Camera intrinsics
    f_mm        : float = 8.0    # focal length (mm)
    sensor_w_mm : float = 6.17  # sensor width  (mm)  — 1/2.9" ≈ 6.17 mm
    sensor_h_mm : float = 4.55  # sensor height (mm)
    img_w_px    : int   = 640   # working resolution width
    img_h_px    : int   = 480   # working resolution height

    # Eye model
    eye_radius_mm : float = 12.0  # human eye radius (~12 mm)
    camera_z_mm   : float = 60.0  # camera-to-eye distance (mm)

    # Derived
    @property
    def px_size_x(self): return self.sensor_w_mm / self.img_w_px
    @property
    def px_size_y(self): return self.sensor_h_mm / self.img_h_px
    @property
    def cx(self): return self.img_w_px / 2.0   # principal point x
    @property
    def cy(self): return self.img_h_px / 2.0   # principal point y
    @property
    def fx_px(self): return self.f_mm / self.px_size_x
    @property
    def fy_px(self): return self.f_mm / self.px_size_y
    @property
    def eye_radius_px(self):
        """Projected eye radius on the image plane (pixels)."""
        return self.fx_px * (self.eye_radius_mm / self.camera_z_mm)


PARAMS = EyeTrackingParams()


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Physics:  pixel  →  mm  →  3-D ray  →  sphere intersection
# ══════════════════════════════════════════════════════════════════════════════

def pixel_to_mm(u: float, v: float, p: EyeTrackingParams):
    """Convert pixel (u, v) to mm offset from the optical axis."""
    return (u - p.cx) * p.px_size_x, (v - p.cy) * p.px_size_y


def ray_sphere_intersect(O: np.ndarray, d: np.ndarray,
                          C: np.ndarray, r: float):
    """
    Ray-sphere intersection.
    Solves  (d·d)t² + 2(d·(O−C))t + (‖O−C‖²−r²) = 0
    Returns (t1, t2) with t1 ≤ t2, or (None, None) if no real roots.
    """
    oc   = O - C
    a    = np.dot(d, d)
    b    = 2.0 * np.dot(d, oc)
    c    = np.dot(oc, oc) - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return None, None
    sq = np.sqrt(disc)
    return (-b - sq) / (2 * a), (-b + sq) / (2 * a)


def pupil_2d_to_3d(u: float, v: float, p: EyeTrackingParams):
    """
    Given 2-D pupil pixel (u, v), return:
      point_3d -- 3-D surface point on the eye sphere (mm, eye-centred)
      gaze_vec -- unit gaze direction vector (eye centre -> surface point)
    Returns (None, None) if the ray misses the sphere.
    """
    x_mm, y_mm = pixel_to_mm(u, v, p)

    # Camera optical centre in eye-centred frame
    O = np.array([0.0, 0.0, p.camera_z_mm])

    # Ray direction: from camera through the image-plane point toward the eye
    d = np.array([x_mm, y_mm, -p.f_mm])
    d = d / np.linalg.norm(d)

    # Eye sphere centred at the origin
    C = np.zeros(3)

    t1, t2 = ray_sphere_intersect(O, d, C, p.eye_radius_mm)
    if t1 is None:
        return None, None

    # Pick the front-facing intersection (smallest positive t)
    t = t1 if t1 > 0 else t2
    if t <= 0:
        return None, None

    point_3d = O + t * d
    gaze_vec  = point_3d / np.linalg.norm(point_3d)   # C is at origin
    return point_3d, gaze_vec


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Pupil detector  —  simple, self-contained dark-blob approach
# ══════════════════════════════════════════════════════════════════════════════

def detect_pupil(frame: np.ndarray):
    """
    Detect the pupil in a BGR frame.
    Returns (ellipse, center_x, center_y) or (None, None, None).

    Strategy:
      1. Find the darkest region of the image (coarse grid scan).
      2. Threshold around that dark region at three levels.
      3. For each threshold pick the largest plausible contour.
      4. Score each candidate by how well it fits an ellipse.
      5. Return the best-scoring ellipse.
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Step 1: locate darkest region (coarse grid scan)
    IGNORE, SKIP, AREA = 20, 10, 20
    h, w = blurred.shape
    min_sum, dark_pt = float('inf'), None
    for y in range(IGNORE, h - IGNORE, SKIP):
        for x in range(IGNORE, w - IGNORE, SKIP):
            patch = blurred[y:y+AREA, x:x+AREA]
            s = int(patch.sum())
            if s < min_sum:
                min_sum = s
                dark_pt = (x + AREA // 2, y + AREA // 2)

    if dark_pt is None:
        return None, None, None

    dark_val = int(blurred[dark_pt[1], dark_pt[0]])

    # Step 2: try three threshold levels
    kernel = np.ones((5, 5), np.uint8)
    best_ellipse, best_score = None, 0.0

    for offset in (10, 20, 35):
        thresh = dark_val + offset
        _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY_INV)

        # Restrict to square around dark point
        mask = np.zeros_like(binary)
        half = 130
        y1 = max(0, dark_pt[1] - half);  y2 = min(h, dark_pt[1] + half)
        x1 = max(0, dark_pt[0] - half);  x2 = min(w, dark_pt[0] + half)
        mask[y1:y2, x1:x2] = 255
        binary = cv2.bitwise_and(binary, mask)

        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # Step 3: filter contours — keep largest non-elongated blob
        best_c, best_a = None, 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 600:
                continue
            rx, ry, rw, rh = cv2.boundingRect(c)
            if max(rw, rh) / max(min(rw, rh), 1) > 3.0:
                continue
            if area > best_a:
                best_a = area
                best_c = c

        if best_c is None or len(best_c) < 5:
            continue

        # Step 4: fit ellipse and score it
        try:
            ellipse = cv2.fitEllipse(best_c)
        except cv2.error:
            continue

        (ex, ey), (ew, eh), _ = ellipse
        if ew <= 0 or eh <= 0 or not np.isfinite(ex) or not np.isfinite(ey):
            continue

        emask = np.zeros_like(dilated)
        cv2.ellipse(emask, ellipse, 255, -1)
        e_area = np.sum(emask == 255)
        if e_area == 0:
            continue

        covered = np.sum((dilated == 255) & (emask == 255))
        score   = covered / e_area

        if score > best_score:
            best_score   = score
            best_ellipse = ellipse

    if best_ellipse is None:
        return None, None, None

    cx, cy = int(best_ellipse[0][0]), int(best_ellipse[0][1])
    return best_ellipse, cx, cy


# ══════════════════════════════════════════════════════════════════════════════
#  4.  Eye-centre estimator  —  rolling average of pupil positions
# ══════════════════════════════════════════════════════════════════════════════

class EyeCentreEstimator:
    """
    The eyeball centre is relatively stable while the pupil moves around it.
    A long rolling average of pupil detections converges to a good 2-D
    estimate of the eye centre without needing any extra geometry.
    """
    def __init__(self, history: int = 200):
        self._hist    = []
        self._maxhist = history

    def update(self, px: int, py: int):
        self._hist.append((px, py))
        if len(self._hist) > self._maxhist:
            self._hist.pop(0)

    def get(self, fallback):
        if not self._hist:
            return fallback
        return (int(np.mean([p[0] for p in self._hist])),
                int(np.mean([p[1] for p in self._hist])))


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Matplotlib 3-D gaze sphere  —  off-screen, converted to cv2 image
# ══════════════════════════════════════════════════════════════════════════════

class GazeSphere3D:
    """
    Off-screen matplotlib figure showing:
      - wireframe unit sphere
      - live gaze direction arrow (yellow)
      - fading history trail (orange)
      - mm-space coordinate annotation
    """

    def __init__(self, size: int = 480):
        self.fig = plt.figure(figsize=(size/100, size/100), dpi=100,
                              facecolor="#0a0a0f")
        self.ax  = self.fig.add_subplot(111, projection="3d",
                                         facecolor="#0a0a0f")
        self._draw_static()
        self._dynamic = []
        self.history  = []
        self.max_hist = 60

    def _draw_static(self):
        ax = self.ax
        u = np.linspace(0, 2*np.pi, 24)
        v = np.linspace(0,   np.pi, 16)
        ax.plot_wireframe(
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
            color="#1a2a4a", linewidth=0.4, alpha=0.5)
        t = np.linspace(0, 2*np.pi, 120)
        ax.plot(np.cos(t), np.sin(t), np.zeros_like(t),
                color="#00aaff", linewidth=1.2, alpha=0.7)
        for vec, col, lbl in [([1.3,0,0], "#ff4466", "X"),
                               ([0,1.3,0], "#44ff88", "Y"),
                               ([0,0,1.3], "#4488ff", "Z")]:
            ax.quiver(0, 0, 0, *vec, color=col, linewidth=1.5,
                      arrow_length_ratio=0.15)
            ax.text(*[x*1.55 for x in vec], lbl, color=col,
                    fontsize=8, fontweight="bold", ha="center", va="center")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_zlim(-1.6, 1.6)
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")
        ax.set_title("Gaze Vector  (unit sphere)",
                     color="#aaccff", fontsize=9, pad=4,
                     fontfamily="monospace")

    def update(self, point_3d: np.ndarray, gaze_vec: np.ndarray):
        for a in self._dynamic:
            try: a.remove()
            except Exception: pass
        self._dynamic = []

        gx, gy, gz = float(gaze_vec[0]), float(gaze_vec[1]), float(gaze_vec[2])

        q = self.ax.quiver(0, 0, 0, gx, gy, gz,
                           color="#ffdd00", linewidth=2.5,
                           arrow_length_ratio=0.18, zorder=10)
        s = self.ax.scatter([gx], [gy], [gz],
                            color="#ff8800", s=45, zorder=11, depthshade=False)
        self._dynamic.extend([q, s])

        self.history.append((gx, gy, gz))
        if len(self.history) > self.max_hist:
            self.history.pop(0)
        n = len(self.history)
        if n > 1:
            hx = [p[0] for p in self.history]
            hy = [p[1] for p in self.history]
            hz = [p[2] for p in self.history]
            for i in range(1, n):
                alpha = 0.08 + 0.6 * (i / n)
                line, = self.ax.plot(hx[i-1:i+1], hy[i-1:i+1], hz[i-1:i+1],
                                     color="#ff8800", alpha=alpha, linewidth=1.2)
                self._dynamic.append(line)

        txt = self.ax.text2D(
            0.02, 0.02,
            f"({point_3d[0]:+.1f}, {point_3d[1]:+.1f}, {point_3d[2]:+.1f}) mm",
            transform=self.ax.transAxes, color="#aaffcc",
            fontsize=7, fontfamily="monospace")
        self._dynamic.append(txt)

        self.fig.canvas.draw()

    def to_cv2(self) -> np.ndarray:
        self.fig.canvas.draw()
        cw, ch = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(ch, cw, 4)
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


# ══════════════════════════════════════════════════════════════════════════════
#  6.  CV frame drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def draw_cv_overlays(frame, ellipse, pupil_x, pupil_y,
                     eye_cx, eye_cy, point_3d, gaze_vec, params):
    """Annotate the CV frame in-place."""
    H, W = frame.shape[:2]

    # Eye-sphere circle + centre dot
    r_px = max(1, int(params.eye_radius_px))
    cv2.circle(frame, (eye_cx, eye_cy), r_px, (255, 50, 50), 2)
    cv2.circle(frame, (eye_cx, eye_cy),    8, (255, 255,  0), -1)

    # Pupil ellipse
    if ellipse is not None:
        (ex, ey), (ew, eh), _ = ellipse
        if ew > 0 and eh > 0 and np.isfinite(ex) and np.isfinite(ey):
            cv2.ellipse(frame, ellipse, (20, 255, 255), 2)

    # Eye-centre to pupil line + extended gaze ray arrow
    if pupil_x is not None and pupil_y is not None:
        cv2.line(frame, (eye_cx, eye_cy), (pupil_x, pupil_y),
                 (255, 150, 50), 2)
        dx  = pupil_x - eye_cx
        dy  = pupil_y - eye_cy
        ex2 = int(eye_cx + 2.2 * dx)
        ey2 = int(eye_cy + 2.2 * dy)
        cv2.arrowedLine(frame, (pupil_x, pupil_y), (ex2, ey2),
                        (80, 255, 80), 3, tipLength=0.22)
        cv2.circle(frame, (pupil_x, pupil_y), 5, (0, 200, 255), -1)

    # HUD bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H - 100), (W, H), (5, 8, 18), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, H - 100), (W, H - 100), (0, 120, 220), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if point_3d is not None and gaze_vec is not None:
        hud = [
            (f"3D pupil : ({point_3d[0]:+6.2f}, {point_3d[1]:+6.2f},"
             f" {point_3d[2]:+6.2f}) mm",        (0, 255, 180)),
            (f"Gaze dir : ({gaze_vec[0]:+.4f}, {gaze_vec[1]:+.4f},"
             f" {gaze_vec[2]:+.4f})",             (255, 220,   0)),
            (f"Pupil 2D : ({pupil_x:3d}, {pupil_y:3d}) px   "
             f"Eye ctr 2D : ({eye_cx}, {eye_cy}) px",
                                                  (160, 200, 255)),
        ]
        for i, (txt, col) in enumerate(hud):
            yp = H - 78 + i * 26
            cv2.putText(frame, txt, (11, yp+1), font, 0.44, (0,   0,   0), 2)
            cv2.putText(frame, txt, (10, yp),   font, 0.44, col,          1)


# ══════════════════════════════════════════════════════════════════════════════
#  7.  Main loop
# ══════════════════════════════════════════════════════════════════════════════

def run(source=0):
    params = PARAMS
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  params.img_w_px)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, params.img_h_px)
    if source == 0:
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    sphere_vis   = GazeSphere3D(size=480)
    eye_est      = EyeCentreEstimator(history=200)
    fallback_ctr = (params.img_w_px // 2, params.img_h_px // 2)

    paused      = False
    debug       = False
    frame_count = 0

    print("\n+========================================+")
    print("|   Eye Tracker 3D  --  Clean Pipeline   |")
    print("+========================================+")
    print("|  SPACE = pause / resume                |")
    print("|  D     = toggle debug window           |")
    print("|  Q     = quit                          |")
    print("+========================================+\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_count += 1
            frame = cv2.resize(frame, (params.img_w_px, params.img_h_px))

            # Pupil detection
            ellipse, pupil_x, pupil_y = detect_pupil(frame)

            # Eye-centre estimate (rolling average)
            if pupil_x is not None:
                eye_est.update(pupil_x, pupil_y)
            eye_cx, eye_cy = eye_est.get(fallback_ctr)

            # 2-D pixel -> 3-D mm via ray-sphere physics
            point_3d, gaze_vec = None, None
            if pupil_x is not None:
                point_3d, gaze_vec = pupil_2d_to_3d(pupil_x, pupil_y, params)

            # Draw overlays on CV frame
            draw_cv_overlays(frame, ellipse, pupil_x, pupil_y,
                             eye_cx, eye_cy, point_3d, gaze_vec, params)

            # Update 3-D sphere every 2nd frame
            if gaze_vec is not None and frame_count % 2 == 0:
                sphere_vis.update(point_3d, gaze_vec)
            sphere_img = cv2.resize(sphere_vis.to_cv2(),
                                    (params.img_w_px, params.img_h_px))

            # Side-by-side composite
            divider  = np.full((params.img_h_px, 4, 3), (30, 60, 100),
                               dtype=np.uint8)
            combined = np.hstack([frame, divider, sphere_img])

            title = np.zeros((36, combined.shape[1], 3), dtype=np.uint8)
            title[:] = (10, 14, 28)
            cv2.putText(title,
                        "EYE TRACKER 3D  |  CV Feed  (mm-space ray-sphere)"
                        "                        Gaze Sphere",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (100, 180, 255), 1)
            cv2.line(title, (0, 35), (title.shape[1], 35), (0, 100, 200), 1)

            cv2.imshow("Eye Tracker 3D", np.vstack([title, combined]))

            # Debug window
            if debug:
                dbg = frame.copy()
                if ellipse is not None:
                    (ex, ey), (ew, eh), _ = ellipse
                    if ew > 0 and eh > 0:
                        cv2.ellipse(dbg, ellipse, (0, 255, 0), 1)
                cv2.imshow("Debug -- pupil ellipse", dbg)

            # Terminal log every 20 frames
            if frame_count % 20 == 0 and point_3d is not None:
                print(f"[{frame_count:05d}]  "
                      f"Pupil=({pupil_x:3d},{pupil_y:3d})  "
                      f"3D=({point_3d[0]:+6.2f},{point_3d[1]:+6.2f},"
                      f"{point_3d[2]:+6.2f}) mm  "
                      f"Gaze=({gaze_vec[0]:+.3f},{gaze_vec[1]:+.3f},"
                      f"{gaze_vec[2]:+.3f})")

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("[PAUSED]" if paused else "[RESUMED]")
        elif key == ord('d'):
            debug = not debug
            if not debug:
                cv2.destroyWindow("Debug -- pupil ellipse")
            print(f"[DEBUG {'ON' if debug else 'OFF'}]")

    cap.release()
    cv2.destroyAllWindows()
    plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    try:
        src = int(src)
    except (ValueError, TypeError):
        pass
    run(src)
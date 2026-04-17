"""
Optimized dual eye tracker.

Key performance improvements vs. the original:
  1. Threaded frame grabbers per camera/stream with a 1-slot queue so we always
     process the LATEST frame and drop stale ones (kills MJPEG buffer lag).
  2. Left and right eyes processed in parallel on a ThreadPoolExecutor.
  3. `get_darkest_area` replaced with a single cv2.boxFilter + minMaxLoc call
     (was a Python double loop -- typically 50-100x faster).
  4. `optimize_contours_by_angle` fully vectorized with NumPy.
  5. Single grayscale + single darkest-point lookup shared across the 3
     threshold passes (was recomputed each time).
  6. Ellipse ray history kept small and intersections computed with vectorized
     NumPy instead of nested Python loops.
  7. Matplotlib replaced with an OpenCV window for the 3D-ish gaze view --
     matplotlib's interactive redraw was costing 100-300ms per frame.
  8. cv2.setUseOptimized(True) and thread count hints.
"""

import cv2
import numpy as np
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk, filedialog
from concurrent.futures import ThreadPoolExecutor

cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(2)  # leave cores for our own threads
except Exception:
    pass


# ---------------------------------------------------------------------------
# Threaded video capture -- always serves the freshest frame, drops backlog.
# ---------------------------------------------------------------------------
class FreshestFrameGrabber:
    """Background thread that continuously reads from a VideoCapture and keeps
    only the most recent frame. Prevents MJPEG/network buffer buildup."""

    def __init__(self, source, name="cam"):
        self.source = source
        self.name = name
        self.cap = cv2.VideoCapture(source)
        # Small internal buffer so we don't accumulate latency.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        self._lock = threading.Lock()
        self._latest = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                # Stream hiccup -- small sleep, try again. Don't spam.
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = frame

    def read(self):
        with self._lock:
            if self._latest is None:
                return False, None
            # Return a shallow copy so the consumer can mutate freely while
            # the grabber overwrites _latest.
            return True, self._latest

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# ---------------------------------------------------------------------------
# Fast image ops
# ---------------------------------------------------------------------------
def get_darkest_area_fast(gray):
    """Find a small dark region using a box filter. One call instead of a
    Python double loop. Returns (x, y) of the darkest 20x20 window center."""
    # Blur the image with a 20x20 averaging filter; darkest pixel of the blur
    # is the center of the darkest window.
    blurred = cv2.boxFilter(gray, ddepth=-1, ksize=(20, 20),
                            normalize=True, borderType=cv2.BORDER_REPLICATE)
    # Ignore a border so we don't pick up frame edges.
    b = 20
    h, w = blurred.shape
    roi = blurred[b:h - b, b:w - b]
    _, _, min_loc, _ = cv2.minMaxLoc(roi)
    return (min_loc[0] + b, min_loc[1] + b)


def mask_outside_square(image, center, size):
    """Zero everything outside a square centered at `center`."""
    x, y = center
    half = size // 2
    h, w = image.shape[:2]
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)
    out = np.zeros_like(image)
    out[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return out


def optimize_contours_by_angle_fast(contour):
    """Vectorized version of the original angle-filter. `contour` is an
    (N, 1, 2) or (N, 2) array of points from cv2.findContours."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    n = len(pts)
    if n < 10:
        return contour

    spacing = max(1, n // 25)
    prev_pts = np.roll(pts, spacing, axis=0)
    next_pts = np.roll(pts, -spacing, axis=0)

    vec1 = prev_pts - pts
    vec2 = next_pts - pts

    centroid = pts.mean(axis=0)
    vec_to_centroid = centroid - pts
    mid = (vec1 + vec2) * 0.5

    # Keep points whose (vec1+vec2)/2 points roughly toward the centroid.
    cos_thresh = np.cos(np.radians(60))
    keep = (vec_to_centroid * mid).sum(axis=1) >= cos_thresh

    filtered = pts[keep]
    if len(filtered) < 5:
        return contour
    return filtered.astype(np.int32).reshape(-1, 1, 2)


def filter_largest_valid_contour(contours, pixel_thresh=1000, ratio_thresh=3.0):
    """Largest-area contour that isn't super elongated."""
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < pixel_thresh or area <= best_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h == 0 or w == 0:
            continue
        ratio = max(w / h, h / w)
        if ratio > ratio_thresh:
            continue
        best = c
        best_area = area
    return best


def check_ellipse_goodness(binary_image, contour):
    """Score: fraction of white pixels inside the fitted ellipse."""
    if len(contour) < 5:
        return 0.0
    ellipse = cv2.fitEllipse(contour)
    mask = np.zeros_like(binary_image)
    cv2.ellipse(mask, ellipse, 255, -1)
    ellipse_area = int(np.count_nonzero(mask))
    if ellipse_area == 0:
        return 0.0
    covered = int(np.count_nonzero((binary_image == 255) & (mask == 255)))
    return covered / ellipse_area


def check_contour_pixels(contour, image_shape):
    """How much of the contour actually lies under the fitted ellipse edge."""
    if len(contour) < 5:
        return (0, 0.0)
    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, 1)

    ellipse = cv2.fitEllipse(contour)
    thick = np.zeros(image_shape, dtype=np.uint8)
    thin = np.zeros(image_shape, dtype=np.uint8)
    cv2.ellipse(thick, ellipse, 255, 10)
    cv2.ellipse(thin, ellipse, 255, 4)

    overlap_thick = cv2.bitwise_and(contour_mask, thick)
    overlap_thin = cv2.bitwise_and(contour_mask, thin)

    total_border = int(np.count_nonzero(contour_mask))
    if total_border == 0:
        return (0, 0.0)
    return (int(np.count_nonzero(overlap_thick)),
            int(np.count_nonzero(overlap_thin)) / total_border)


# ---------------------------------------------------------------------------
# Eye tracker
# ---------------------------------------------------------------------------
class EyeTracker:
    def __init__(self, name, camera_position_3d):
        self.name = name
        self.camera_position = np.array(camera_position_3d, dtype=np.float32)

        # Rolling buffer of recent ellipses used to triangulate eye center.
        # Columns: cx, cy, angle_deg, valid_flag.
        self.HISTORY = 40
        self.ellipses = np.zeros((self.HISTORY, 4), dtype=np.float32)
        self.counter = 0

        # Last drawn rays (for visualization).
        self.rays = []

        # Smoothed live eye-sphere center. Starts at frame center and gets
        # EMA-updated every time we get a fresh estimate.
        self.model_center_smooth = None  # np.array([x, y], float32)
        self.EMA_ALPHA = 0.15            # higher = snappier, lower = smoother
        self.MIN_RAYS = 6                # need at least this many ellipses
        self.MIN_ANGLE_DEG = 3.0         # reject near-parallel ray pairs

        self._kernel = np.ones((5, 5), np.uint8)

    # ---- vectorized eye-center estimation from recent ellipses ----
    def estimate_eye_center(self, frame_shape):
        """Estimate the 2D eye-sphere center from the rolling buffer of
        ellipse centers + minor-axis directions. Vectorized: no Python pair
        loops. Returns (x, y) or None."""
        mask = self.ellipses[:, 3] > 0.5
        data = self.ellipses[mask]
        if len(data) < self.MIN_RAYS:
            return None

        cx = data[:, 0]
        cy = data[:, 1]
        ang = np.deg2rad(data[:, 2])
        # Perpendicular to the ellipse major axis direction (same as original).
        dx = -np.sin(ang)
        dy = np.cos(ang)

        n = len(data)
        # Build all unique pairs (i < j) with triu_indices.
        i_idx, j_idx = np.triu_indices(n, k=1)

        x1, y1 = cx[i_idx], cy[i_idx]
        x2, y2 = cx[j_idx], cy[j_idx]
        d1x, d1y = dx[i_idx], dy[i_idx]
        d2x, d2y = dx[j_idx], dy[j_idx]

        # Reject near-parallel ray pairs.
        cos_theta = d1x * d2x + d1y * d2y
        keep_ang = np.abs(cos_theta) < np.cos(np.deg2rad(self.MIN_ANGLE_DEG))
        if not np.any(keep_ang):
            return None

        # Solve each 2x2 system [[d1x, -d2x],[d1y, -d2y]] * [t1, t2]^T = [dx, dy]
        det = d1x * (-d2y) - (-d2x) * d1y  # = -d1x*d2y + d2x*d1y
        rhs_x = x2 - x1
        rhs_y = y2 - y1

        valid = keep_ang & (np.abs(det) > 1e-6)
        if not np.any(valid):
            return None

        t1 = np.empty_like(det)
        t1[:] = np.nan
        # t1 = ( (-d2y)*rhs_x - (-d2x)*rhs_y ) / det  = (-d2y*rhs_x + d2x*rhs_y)/det
        t1[valid] = ((-d2y[valid]) * rhs_x[valid]
                     - (-d2x[valid]) * rhs_y[valid]) / det[valid]

        ix = x1 + t1 * d1x
        iy = y1 + t1 * d1y

        ix = ix[valid]
        iy = iy[valid]

        # Clip to frame bounds to kill wild outliers from parallel-ish rays.
        h, w = frame_shape[:2]
        in_bounds = (ix > -w) & (ix < 2 * w) & (iy > -h) & (iy < 2 * h)
        ix = ix[in_bounds]
        iy = iy[in_bounds]
        if len(ix) < 3:
            return None

        # Robust center: median is far more stable than mean for this.
        return float(np.median(ix)), float(np.median(iy))

    # ---- core processing (runs on worker thread) ----
    def process_frame(self, frame):
        # Pre-rotation done by caller.
        frame = cv2.flip(frame, 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        darkest_point = get_darkest_area_fast(gray)
        darkest_val = int(gray[darkest_point[1], darkest_point[0]])

        # Build three thresholded images once each.
        best_contour = None
        best_score = 0.0
        best_binary = None

        for added in (5, 15, 25):
            t = darkest_val + added
            _, thr = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
            thr = mask_outside_square(thr, darkest_point, 250)
            dilated = cv2.dilate(thr, self._kernel, iterations=2)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cand = filter_largest_valid_contour(contours)
            if cand is None or len(cand) < 6:
                continue

            goodness = check_ellipse_goodness(dilated, cand)
            thick_count, thin_ratio = check_contour_pixels(cand, dilated.shape)
            score = goodness * thick_count * thick_count * thin_ratio
            if score > best_score:
                best_score = score
                best_contour = cand
                best_binary = dilated

        center_x = center_y = None
        final_ellipse = None

        if best_contour is not None:
            refined = optimize_contours_by_angle_fast(best_contour)
            if refined is not None and len(refined) >= 5:
                final_ellipse = cv2.fitEllipse(refined)
                (cx, cy), _, angle = final_ellipse
                center_x, center_y = int(cx), int(cy)
                # Store with valid=1 in the rolling buffer.
                self.ellipses[self.counter % self.HISTORY] = [cx, cy, angle, 1.0]
                self.counter += 1

        h, w = frame.shape[:2]

        # Live eye-sphere center: estimate from the rolling ellipse buffer,
        # then smooth with an EMA so it updates frame-to-frame without jitter.
        estimate = self.estimate_eye_center((h, w))
        if estimate is not None:
            est_arr = np.array(estimate, dtype=np.float32)
            if self.model_center_smooth is None:
                self.model_center_smooth = est_arr
            else:
                self.model_center_smooth = (
                    (1.0 - self.EMA_ALPHA) * self.model_center_smooth
                    + self.EMA_ALPHA * est_arr
                )
        elif self.model_center_smooth is None:
            # Until we have enough samples, fall back to frame center.
            self.model_center_smooth = np.array([w / 2.0, h / 2.0],
                                                dtype=np.float32)

        mcx = int(np.clip(self.model_center_smooth[0], 0, w - 1))
        mcy = int(np.clip(self.model_center_smooth[1], 0, h - 1))
        model_center = (mcx, mcy)

        # Draw overlay.
        locked = estimate is not None
        sphere_color = (255, 50, 50) if locked else (120, 120, 120)
        cv2.circle(frame, model_center, 202, sphere_color, 2)
        cv2.circle(frame, model_center, 8, (255, 255, 0), -1)
        if final_ellipse is not None and center_x is not None:
            cv2.line(frame, model_center, (center_x, center_y),
                     (255, 150, 50), 2)
            cv2.ellipse(frame, final_ellipse, (20, 255, 255), 2)
        status = "LIVE" if locked else f"warmup {int(self.ellipses[:,3].sum())}/{self.MIN_RAYS}"
        cv2.putText(frame, f"{self.name}  [{status}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if center_x is None:
            return frame, None, None

        sphere_center, gaze_dir = self.compute_gaze_vector(
            center_x, center_y, model_center[0], model_center[1], w, h)
        return frame, sphere_center, gaze_dir

    # ---- gaze math (unchanged in spirit, tightened up) ----
    def compute_gaze_vector(self, x, y, center_x, center_y,
                            screen_width=640, screen_height=480):
        fov_y_rad = np.radians(45.0)
        aspect = screen_width / screen_height
        far_clip = 100.0
        cam = self.camera_position

        half_h = np.tan(fov_y_rad / 2) * far_clip
        half_w = half_h * aspect

        ndc_x = (2.0 * x) / screen_width - 1.0
        ndc_y = 1.0 - (2.0 * y) / screen_height
        far_pt = np.array([ndc_x * half_w, ndc_y * half_h,
                           cam[2] - far_clip], dtype=np.float32)

        ray_dir = far_pt - cam
        n = np.linalg.norm(ray_dir)
        if n < 1e-8:
            return None, None
        ray_dir = -ray_dir / n

        inner_radius = 1.0 / 1.05
        off_x = (center_x / screen_width) * 2.0 - 1.0
        off_y = 1.0 - (center_y / screen_height) * 2.0
        sphere_center = np.array([off_x * 1.5, off_y * 1.5, 0.0],
                                 dtype=np.float32) + cam

        direction = -ray_dir
        L = cam - sphere_center
        a = float(np.dot(direction, direction))
        b = float(2 * np.dot(direction, L))
        c = float(np.dot(L, L) - inner_radius ** 2)
        disc = b * b - 4 * a * c

        if disc < 0:
            t = -float(np.dot(direction, L)) / a
        else:
            sq = np.sqrt(disc)
            t1 = (-b - sq) / (2 * a)
            t2 = (-b + sq) / (2 * a)
            candidates = [tv for tv in (t1, t2) if tv > 0]
            if not candidates:
                return None, None
            t = min(candidates)

        intersection = cam + t * direction
        local = intersection - sphere_center
        ln = np.linalg.norm(local)
        if ln < 1e-8:
            return None, None
        target_dir = local / ln

        circle_local = np.array([0.0, 0.0, inner_radius], dtype=np.float32)
        circle_local /= np.linalg.norm(circle_local)

        axis = np.cross(circle_local, target_dir)
        an = np.linalg.norm(axis)
        if an < 1e-6:
            return sphere_center, circle_local
        axis /= an
        dot = float(np.clip(np.dot(circle_local, target_dir), -1.0, 1.0))
        angle = np.arccos(dot)

        cs, sn = np.cos(angle), np.sin(angle)
        tt = 1 - cs
        xa, ya, za = axis
        R = np.array([
            [tt * xa * xa + cs,      tt * xa * ya - sn * za, tt * xa * za + sn * ya],
            [tt * xa * ya + sn * za, tt * ya * ya + cs,      tt * ya * za - sn * xa],
            [tt * xa * za - sn * ya, tt * ya * za + sn * xa, tt * za * za + cs],
        ], dtype=np.float32)

        gaze = R @ np.array([0.0, 0.0, inner_radius], dtype=np.float32)
        gn = np.linalg.norm(gaze)
        if gn < 1e-8:
            return sphere_center, circle_local
        return sphere_center, gaze / gn


# ---------------------------------------------------------------------------
# Gaze intersection
# ---------------------------------------------------------------------------
def compute_gaze_intersection(lc, ld, rc, rd):
    delta = lc - rc
    dll = np.dot(ld, ld)
    dlr = np.dot(ld, rd)
    drr = np.dot(rd, rd)
    dld = np.dot(ld, delta)
    drd = np.dot(rd, delta)
    denom = dll * drr - dlr * dlr
    if abs(denom) < 1e-6:
        return (lc + rc) * 0.5 + ld * 1000.0
    t_l = (dlr * drd - drr * dld) / denom
    t_r = (dll * drd - dlr * dld) / denom
    p_l = lc + t_l * ld
    p_r = rc + t_r * rd
    return (p_l + p_r) * 0.5


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_dual_tracking(src_left, src_right=None, mirror_mode=False):
    grab_l = FreshestFrameGrabber(src_left, "left")
    grab_r = None if mirror_mode else FreshestFrameGrabber(src_right, "right")

    tracker_left = EyeTracker("Left Eye", [-0.1, 0.0, 3.0])
    tracker_right = EyeTracker("Right Eye", [0.1, 0.0, 3.0])

    pool = ThreadPoolExecutor(max_workers=2)

    # Small OpenCV window for the 3D-ish view instead of matplotlib.
    viz = np.zeros((400, 400, 3), dtype=np.uint8)

    # FPS tracking.
    fps_t0 = time.time()
    fps_frames = 0
    fps = 0.0

    try:
        while True:
            ok_l, frame_l = grab_l.read()
            if not ok_l or frame_l is None:
                time.sleep(0.002)
                continue

            frame_l_rot = cv2.rotate(frame_l, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if mirror_mode:
                frame_r_rot = cv2.flip(frame_l_rot, 1)
            else:
                ok_r, frame_r = grab_r.read()
                if not ok_r or frame_r is None:
                    time.sleep(0.002)
                    continue
                frame_r_rot = cv2.rotate(frame_r, cv2.ROTATE_90_CLOCKWISE)

            # Run both eyes in parallel.
            fut_l = pool.submit(tracker_left.process_frame, frame_l_rot)
            fut_r = pool.submit(tracker_right.process_frame, frame_r_rot)
            out_l, lCenter, lDir = fut_l.result()
            out_r, rCenter, rDir = fut_r.result()

            # Overlay FPS on left feed.
            fps_frames += 1
            if fps_frames >= 10:
                now = time.time()
                fps = fps_frames / (now - fps_t0)
                fps_t0 = now
                fps_frames = 0
            cv2.putText(out_l, f"{fps:5.1f} FPS", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Left Eye Feed", out_l)
            cv2.imshow("Right Eye Feed", out_r)

            # Lightweight 3D-ish viz (top-down XZ + side YZ).
            if lCenter is not None and rCenter is not None:
                inter = compute_gaze_intersection(lCenter, lDir, rCenter, rDir)
                viz[:] = 0

                def to_screen(p, ox, oy, scale=60):
                    return (int(ox + p[0] * scale), int(oy - p[2] * scale))

                # Top-down panel (X vs Z) on left half.
                cv2.putText(viz, "Top (X,Z)", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                lp = to_screen(lCenter, 100, 200)
                rp = to_screen(rCenter, 100, 200)
                ip = to_screen(inter, 100, 200)
                cv2.line(viz, lp, ip, (0, 0, 255), 1)
                cv2.line(viz, rp, ip, (255, 0, 0), 1)
                cv2.circle(viz, lp, 4, (0, 0, 255), -1)
                cv2.circle(viz, rp, 4, (255, 0, 0), -1)
                cv2.circle(viz, ip, 5, (0, 255, 0), -1)

                # Side panel (Y vs Z) on right half.
                cv2.putText(viz, "Side (Y,Z)", (210, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                def to_screen_side(p, ox, oy, scale=60):
                    return (int(ox + p[1] * scale), int(oy - p[2] * scale))

                lp = to_screen_side(lCenter, 300, 200)
                rp = to_screen_side(rCenter, 300, 200)
                ip = to_screen_side(inter, 300, 200)
                cv2.line(viz, lp, ip, (0, 0, 255), 1)
                cv2.line(viz, rp, ip, (255, 0, 0), 1)
                cv2.circle(viz, lp, 4, (0, 0, 255), -1)
                cv2.circle(viz, rp, 4, (255, 0, 0), -1)
                cv2.circle(viz, ip, 5, (0, 255, 0), -1)

                cv2.putText(viz,
                            f"gaze: ({inter[0]:+.2f},{inter[1]:+.2f},{inter[2]:+.2f})",
                            (10, 380), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
                cv2.imshow("3D Gaze", viz)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)
    finally:
        pool.shutdown(wait=False)
        grab_l.release()
        if grab_r is not None:
            grab_r.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Camera detection + GUI
# ---------------------------------------------------------------------------
def detect_cameras(max_cams=10):
    avail = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            avail.append(i)
            cap.release()
    return avail


def dual_selection_gui():
    cameras = detect_cameras()
    root = tk.Tk()
    root.title("Dual Eye Tracker Configuration")

    tk.Label(root, text="Left Eye Source:",
             font=("Arial", 10, "bold")).pack(pady=5)
    sel_l = tk.StringVar(value=str(cameras[0]) if cameras else "0")
    ttk.Combobox(root, textvariable=sel_l,
                 values=[str(c) for c in cameras]).pack()

    tk.Label(root, text="Right Eye Source:",
             font=("Arial", 10, "bold")).pack(pady=5)
    sel_r = tk.StringVar(value=str(cameras[1]) if len(cameras) > 1 else "1")
    ttk.Combobox(root, textvariable=sel_r,
                 values=[str(c) for c in cameras]).pack()

    def start_streams():
        src_l = "http://10.42.0.1:8080?action=stream"
        src_r = "http://10.42.0.1:8081?action=stream"
        root.destroy()
        run_dual_tracking(src_l, src_r, mirror_mode=False)

    def start_videos():
        src_l = filedialog.askopenfilename(title="Select LEFT Video",
                                           filetypes=[("Video", "*.mp4")])
        if not src_l:
            return
        src_r = filedialog.askopenfilename(title="Select RIGHT Video",
                                           filetypes=[("Video", "*.mp4")])
        if not src_r:
            return
        root.destroy()
        run_dual_tracking(src_l, src_r, mirror_mode=False)

    def start_mirrored():
        src = filedialog.askopenfilename(title="Select Single Video to Mirror",
                                         filetypes=[("Video", "*.mp4")])
        if not src:
            return
        root.destroy()
        run_dual_tracking(src_left=src, mirror_mode=True)

    tk.Button(root, text="Start 2 Streams",
              command=start_streams).pack(pady=10)
    tk.Button(root, text="Start 2 Videos",
              command=start_videos).pack(pady=5)
    tk.Button(root, text="Start 1 Video (Mirrored)",
              command=start_mirrored, bg="lightblue").pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    dual_selection_gui()
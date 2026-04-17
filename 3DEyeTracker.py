"""
Dual eye tracker with the OLD (single-eye) eye-center algorithm restored,
but all of the new file's infrastructure kept intact:

  * FreshestFrameGrabber threaded capture (always-latest-frame)
  * Parallel per-eye processing on a ThreadPoolExecutor
  * Fast image ops (boxFilter darkest-area, vectorized contour angle filter,
    shared grayscale + shared darkest point across threshold passes)
  * OpenCV-based 3D-ish gaze viz (no matplotlib)
  * cv2.setUseOptimized, thread hints, FPS overlay, etc.

The eye-center estimation itself is a faithful port of the original
single-eye code:

  * rays built from ellipse angle as (cos, sin) scaled by minor_axis/2
    (i.e. along the major axis), NOT perpendicular to it
  * per frame, pick N=5 random ellipses from a rolling list (cap 100),
    intersect only CONSECUTIVE pairs (4 pairs)
  * reject a pair if |angle1 - angle2| < 2 degrees
  * only keep intersections that lie strictly inside the frame
  * append to a per-eye stored_intersections buffer capped at M=1500,
    take the MEAN of that whole buffer as the raw center
  * additionally smooth via mean-of-last-200 (update_and_average_point)
  * fallback to (320, 240) -> prev_model_center_avg stickiness exactly
    like the original
"""

import cv2
import numpy as np
import threading
import queue  # noqa: F401  (kept for parity with original new file)
import time
import random
import tkinter as tk
from tkinter import ttk, filedialog
from concurrent.futures import ThreadPoolExecutor

cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(2)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Threaded video capture -- unchanged from the new file.
# ---------------------------------------------------------------------------
class FreshestFrameGrabber:
    def __init__(self, source, name="cam"):
        self.source = source
        self.name = name
        self.cap = cv2.VideoCapture(source)
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
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = frame

    def read(self):
        with self._lock:
            if self._latest is None:
                return False, None
            return True, self._latest

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# ---------------------------------------------------------------------------
# Fast image ops -- unchanged from the new file (these are non-algorithmic
# speedups that don't touch the eye-center math).
# ---------------------------------------------------------------------------
def get_darkest_area_fast(gray):
    blurred = cv2.boxFilter(gray, ddepth=-1, ksize=(20, 20),
                            normalize=True, borderType=cv2.BORDER_REPLICATE)
    b = 20
    h, w = blurred.shape
    roi = blurred[b:h - b, b:w - b]
    _, _, min_loc, _ = cv2.minMaxLoc(roi)
    return (min_loc[0] + b, min_loc[1] + b)


def mask_outside_square(image, center, size):
    x, y = center
    half = size // 2
    h, w = image.shape[:2]
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)
    out = np.zeros_like(image)
    out[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return out


def optimize_contours_by_angle_fast(contour):
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

    cos_thresh = np.cos(np.radians(60))
    keep = (vec_to_centroid * mid).sum(axis=1) >= cos_thresh

    filtered = pts[keep]
    if len(filtered) < 5:
        return contour
    return filtered.astype(np.int32).reshape(-1, 1, 2)


def filter_largest_valid_contour(contours, pixel_thresh=1000, ratio_thresh=3.0):
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
# Eye tracker -- eye-center math reverted to the ORIGINAL algorithm.
# ---------------------------------------------------------------------------
class EyeTracker:
    def __init__(self, name, camera_position_3d):
        self.name = name
        self.camera_position = np.array(camera_position_3d, dtype=np.float32)

        # --- Original-algorithm state (formerly module-level globals) ---
        # Rolling list of ellipse tuples ((cx,cy),(maj,min),angle) from
        # recent successful fits. Same role as the old `ray_lines`.
        self.ray_lines = []
        self.max_rays = 100

        # Persistent buffer of past intersection points (the old
        # `stored_intersections`), capped at M=1500.
        self.stored_intersections = []
        self.max_stored_intersections = 1500

        # Rolling list of per-frame averaged centers, used for the
        # last-N smoothing pass (the old `model_centers`, cap 200).
        self.model_centers = []
        self.model_centers_window = 200

        # Hardcoded fallback + "stickiness" of previous output, exactly
        # like the old code. 320x240 happens to match a 640x480 frame
        # center; we keep the magic number for fidelity.
        self.prev_model_center_avg = (320, 240)

        # Algorithm hyperparameters matching the original
        # compute_average_intersection call: N=5, M=1500, spacing=5.
        self.N_RANDOM_LINES = 5
        self.MIN_ANGLE_DIFF_DEG = 2.0

        self._kernel = np.ones((5, 5), np.uint8)

    # -----------------------------------------------------------------
    # Old-algorithm helpers, ported verbatim but as instance methods.
    # -----------------------------------------------------------------
    @staticmethod
    def _find_line_intersection(ellipse1, ellipse2):
        """Intersection of two lines built from ellipses, OLD convention:
        direction = (minor/2)*(cos(angle), sin(angle))  -- along major axis."""
        (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
        (cx2, cy2), (_, minor_axis2), angle2 = ellipse2

        a1 = np.deg2rad(angle1)
        a2 = np.deg2rad(angle2)

        dx1 = (minor_axis1 / 2.0) * np.cos(a1)
        dy1 = (minor_axis1 / 2.0) * np.sin(a1)
        dx2 = (minor_axis2 / 2.0) * np.cos(a2)
        dy2 = (minor_axis2 / 2.0) * np.sin(a2)

        A = np.array([[dx1, -dx2], [dy1, -dy2]])
        B = np.array([cx2 - cx1, cy2 - cy1])

        if np.linalg.det(A) == 0:
            return None

        t1, _ = np.linalg.solve(A, B)
        ix = cx1 + t1 * dx1
        iy = cy1 + t1 * dy1
        return (int(ix), int(iy))

    def _compute_average_intersection(self, frame_shape, N):
        """Port of the old compute_average_intersection.

        * picks N random ellipses from self.ray_lines
        * intersects only consecutive pairs in that random sample
        * skips pairs whose angle differs by < 2 degrees
        * only keeps intersections inside the frame
        * appends to self.stored_intersections (cap 1500)
        * returns MEAN of ALL stored intersections as (int, int), or None
        """
        if len(self.ray_lines) < 2 or N < 2:
            return (0, 0)

        h, w = frame_shape[:2]

        selected = random.sample(self.ray_lines,
                                 min(N, len(self.ray_lines)))

        new_intersections = []
        for i in range(len(selected) - 1):
            line1 = selected[i]
            line2 = selected[i + 1]
            angle1 = line1[2]
            angle2 = line2[2]
            if abs(angle1 - angle2) >= self.MIN_ANGLE_DIFF_DEG:
                pt = self._find_line_intersection(line1, line2)
                if pt is not None and 0 <= pt[0] < w and 0 <= pt[1] < h:
                    new_intersections.append(pt)
                    self.stored_intersections.append(pt)

        # Prune to last M.
        if len(self.stored_intersections) > self.max_stored_intersections:
            self.stored_intersections = \
                self.stored_intersections[-self.max_stored_intersections:]

        if not new_intersections:
            return None

        # MEAN over the entire stored buffer (this is what the old code did --
        # note it uses stored_intersections, not just this frame's adds).
        avg_x = np.mean([p[0] for p in self.stored_intersections])
        avg_y = np.mean([p[1] for p in self.stored_intersections])
        return (int(avg_x), int(avg_y))

    def _update_and_average_point(self, new_point):
        """Port of the old update_and_average_point with window=200."""
        self.model_centers.append(new_point)
        if len(self.model_centers) > self.model_centers_window:
            self.model_centers.pop(0)

        if not self.model_centers:
            return None

        avg_x = int(np.mean([p[0] for p in self.model_centers]))
        avg_y = int(np.mean([p[1] for p in self.model_centers]))
        return (avg_x, avg_y)

    # -----------------------------------------------------------------
    # Core per-frame processing.
    # -----------------------------------------------------------------
    def process_frame(self, frame):
        frame = cv2.flip(frame, 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        darkest_point = get_darkest_area_fast(gray)
        darkest_val = int(gray[darkest_point[1], darkest_point[0]])

        best_contour = None
        best_score = 0.0

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

        center_x = center_y = None
        final_ellipse = None

        if best_contour is not None:
            refined = optimize_contours_by_angle_fast(best_contour)
            if refined is not None and len(refined) >= 5:
                final_ellipse = cv2.fitEllipse(refined)
                (cx, cy), _, _ = final_ellipse
                center_x, center_y = int(cx), int(cy)

                # Append the full ellipse tuple to ray_lines (OLD style --
                # we store ((cx,cy),(maj,min),angle), not just the angle).
                self.ray_lines.append(final_ellipse)
                if len(self.ray_lines) > self.max_rays:
                    self.ray_lines = self.ray_lines[-self.max_rays:]

        h, w = frame.shape[:2]

        # --- OLD eye-center algorithm ---
        model_center_average = (320, 240)
        raw = self._compute_average_intersection((h, w), self.N_RANDOM_LINES)
        if raw is not None:
            smoothed = self._update_and_average_point(raw)
            if smoothed is not None:
                model_center_average = smoothed

        # Preserve the old stickiness: if we fell through to (320,240),
        # use previous; else update previous.
        if model_center_average[0] == 320:
            model_center_average = self.prev_model_center_avg
        if model_center_average[0] != 0:
            self.prev_model_center_avg = model_center_average

        # Draw overlay.
        # Locked == we have at least one real intersection stored.
        locked = len(self.stored_intersections) > 0
        sphere_color = (255, 50, 50) if locked else (120, 120, 120)
        cv2.circle(frame, model_center_average, 202, sphere_color, 2)
        cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)
        if final_ellipse is not None and center_x is not None:
            cv2.line(frame, model_center_average, (center_x, center_y),
                     (255, 150, 50), 2)
            cv2.ellipse(frame, final_ellipse, (20, 255, 255), 2)

        status = ("LIVE" if locked
                  else f"warmup {len(self.ray_lines)}/{self.N_RANDOM_LINES}")
        cv2.putText(frame, f"{self.name}  [{status}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if center_x is None:
            return frame, None, None

        sphere_center, gaze_dir = self.compute_gaze_vector(
            center_x, center_y,
            model_center_average[0], model_center_average[1],
            w, h)
        return frame, sphere_center, gaze_dir

    # -----------------------------------------------------------------
    # Gaze math (unchanged from the new file).
    # -----------------------------------------------------------------
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
# Gaze intersection -- unchanged.
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
# Main loop -- unchanged from the new file.
# ---------------------------------------------------------------------------
def run_dual_tracking(src_left, src_right=None, mirror_mode=False):
    grab_l = FreshestFrameGrabber(src_left, "left")
    grab_r = None if mirror_mode else FreshestFrameGrabber(src_right, "right")

    tracker_left = EyeTracker("Left Eye", [-0.1, 0.0, 3.0])
    tracker_right = EyeTracker("Right Eye", [0.1, 0.0, 3.0])

    pool = ThreadPoolExecutor(max_workers=2)

    viz = np.zeros((400, 400, 3), dtype=np.uint8)

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

            fut_l = pool.submit(tracker_left.process_frame, frame_l_rot)
            fut_r = pool.submit(tracker_right.process_frame, frame_r_rot)
            out_l, lCenter, lDir = fut_l.result()
            out_r, rCenter, rDir = fut_r.result()

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

            if lCenter is not None and rCenter is not None:
                inter = compute_gaze_intersection(lCenter, lDir, rCenter, rDir)
                viz[:] = 0

                def to_screen(p, ox, oy, scale=60):
                    return (int(ox + p[0] * scale), int(oy - p[2] * scale))

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
# Camera detection + GUI -- unchanged.
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
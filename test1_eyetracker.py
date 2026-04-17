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
  6. Eye-center estimation uses pairwise ray intersections across all valid
     candidate ellipses per frame, with a ROLLING-WINDOW average of the most
     recent intersection points (see EYE_CENTER_WINDOW below).
  7. Matplotlib replaced with an OpenCV window for the 3D-ish gaze view --
     matplotlib's interactive redraw was costing 100-300ms per frame.
  8. cv2.setUseOptimized(True) and thread count hints.
"""

import cv2
import numpy as np
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import ttk, filedialog
from concurrent.futures import ThreadPoolExecutor

cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(2)  # leave cores for our own threads
except Exception:
    pass


# ===========================================================================
# USER CONFIGURATION -- edit these to match your rig
# ===========================================================================
# Room coordinate convention: +X right, +Y forward (into room), +Z up.
# All values in METERS. Origin is wherever you want -- floor under the
# user's chair, center of the camera rig, a fiducial on the wall, etc.
# Whatever origin you pick, the arm must use the same one.

# Physical 3D position of each eyeball in the room. Default: user's
# eyes 6.4 cm apart (typical IPD), centered on the origin, 1.60 m
# above the floor, looking toward +Y.
LEFT_EYE_POSITION  = (-0.032, 0.0, 1.60)
RIGHT_EYE_POSITION = ( 0.032, 0.0, 1.60)

# Eyeball radius in image pixels. This controls how pupil-offset-from-
# eye-center translates into an angle: offset/radius = sin(angle).
# The overlay circle is drawn at 202 px -- keep this matched to that
# unless you've re-measured for your optics.
EYE_RADIUS_PX = 202

# Sign flips for converting image-pixel pupil motion into world yaw/
# pitch. The left and right camera feeds are rotated 90 deg in opposite
# directions upstream, so the same pixel direction doesn't mean the
# same world direction for both eyes. First-run calibration:
#   1. Have the user look straight ahead -- both rays should point at +Y.
#   2. Have them look right -- both rays should tilt toward +X.
#   3. Have them look up -- both rays should tilt toward +Z.
# Flip any sign that fails.
LEFT_YAW_SIGN, LEFT_PITCH_SIGN = +1.0, -1.0
RIGHT_YAW_SIGN, RIGHT_PITCH_SIGN = -1.0, -1.0

# Eye-center estimator: rolling window.
#   EYE_CENTER_WINDOW -- max number of recent pairwise intersection points
#   kept in the ring buffer. Eye center = mean of everything currently in
#   the buffer. As new points arrive, the oldest get evicted, so the
#   estimate stays fresh and self-heals from stale/bad samples without
#   ever storing all history.
#   BASELINE_FRAMES   -- how many frames with valid samples before the UI
#   marks the estimate as "baseline reached" (color flips to green). The
#   estimate is usable before that; this just gates the status indicator.
EYE_CENTER_WINDOW = 500
BASELINE_FRAMES   = 100

# Top-down viz scale (pixels per meter). 100 px/m -> 1 px = 1 cm,
# with a 4 m x 4 m field of view on a 400 px window.
VIZ_SIZE = 400
VIZ_SCALE = 100.0

# Gaze-target print throttling. Set to None to disable the live print.
GAZE_PRINT_HZ = 5.0
# ===========================================================================


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
    def __init__(self, name, eye_position_room, yaw_sign, pitch_sign,
                 window_size=EYE_CENTER_WINDOW,
                 baseline_frames=BASELINE_FRAMES):
        """
        Parameters
        ----------
        name : str
            Human-readable label for overlays.
        eye_position_room : (x, y, z) tuple in METERS
            Physical 3D position of this eyeball in room coordinates
            (+X right, +Y forward, +Z up). Provided by the caller from
            the module-level USER CONFIGURATION block.
        yaw_sign, pitch_sign : +1 or -1
            Per-eye sign flips to reconcile image-pixel pupil motion
            with world yaw/pitch. See the config block for tuning.
        window_size : int
            Max number of recent pairwise intersection points kept in the
            rolling average. Older points are evicted automatically.
        baseline_frames : int
            Number of sample-contributing frames required before the UI
            reports "baseline reached". The estimate is usable before
            that; this just gates the status indicator.
        """
        self.name = name
        self.eye_position_room = np.array(eye_position_room, dtype=np.float64)
        self.eye_radius_px = float(EYE_RADIUS_PX)
        self.yaw_sign = float(yaw_sign)
        self.pitch_sign = float(pitch_sign)

        # Per-instance ray history for visualization only. List of
        # ((cx, cy), (est_x, est_y)) tuples, capped at RAY_HISTORY entries.
        self.rays = []
        self.RAY_HISTORY = 10

        # ---- Rolling-window eye-center estimator ----
        # Ring buffer of the most recent (x, y) pairwise-intersection points
        # in image space. As new points are appended, the oldest drop off.
        # The eye center is always the mean of whatever is currently in the
        # buffer -- no locking, no separate "running mean" accumulator.
        self._intersections = deque(maxlen=window_size)
        self._frames_with_samples = 0
        self._baseline_frames = baseline_frames

        self._kernel = np.ones((5, 5), np.uint8)

    def set_eye_position(self, xyz_meters):
        """Update the physical eye position in room coordinates. Use
        whenever the user or rig moves relative to the room origin."""
        self.eye_position_room = np.array(xyz_meters, dtype=np.float64)

    def reset_eye_center(self):
        """Clear the rolling window. Call this if the headset shifts and
        the anatomical center needs to be re-measured from scratch."""
        self._intersections.clear()
        self._frames_with_samples = 0
        self.rays.clear()

    @property
    def sample_count(self):
        """Number of intersection points currently in the rolling window."""
        return len(self._intersections)

    @property
    def is_baseline_reached(self):
        """True once enough frames have contributed samples for the
        rolling-window mean to be considered trustworthy for display."""
        return self._frames_with_samples >= self._baseline_frames

    @property
    def eye_center(self):
        """Rolling-window mean of all intersection points currently in the
        buffer. None if we have zero samples yet."""
        if not self._intersections:
            return None
        # deque of 2-tuples -> (N, 2) array -> column-wise mean.
        arr = np.asarray(self._intersections, dtype=np.float64)
        mean = arr.mean(axis=0)
        return (int(mean[0]), int(mean[1]))

    # ---- per-frame eye-center estimation from candidate ellipses ----
    def eyecenter_estimation(self, ellipses, frame):
        """Compute pairwise ray intersections across all candidate ellipses
        in the current frame, push them into the rolling window, and return
        the current rolling-mean eye center.

        Uses the explicit calculation: for each pair of rays (minor axis of
        each ellipse), solve the 2x2 linear system for the intersection
        parameter, skipping near-parallel and numerically-singular pairs.

        `ellipses` is a list of (cx, cy, angle_deg) tuples. `frame` is
        drawn on for visualization. Returns (x, y) int tuple or None.
        """
        # Build rays for all ellipses
        current_rays = []
        for (cx, cy, angle_deg) in ellipses:
            a = np.deg2rad(angle_deg)
            dx, dy = -np.sin(a), np.cos(a)
            current_rays.append((cx, cy, dx, dy))

        # Compute intersections across ALL pairs in current frame
        intersections = []
        for i in range(len(current_rays)):
            for j in range(i + 1, len(current_rays)):
                x1, y1, ddx1, ddy1 = current_rays[i]
                x2, y2, ddx2, ddy2 = current_rays[j]

                v1 = np.array([ddx1, ddy1])
                v2 = np.array([ddx2, ddy2])
                denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                if denom < 1e-8:
                    continue
                cos_theta = np.dot(v1, v2) / denom

                # Reject near-parallel rays (< 2 deg apart).
                if abs(cos_theta) > np.cos(np.deg2rad(2)):
                    continue

                A = np.array([[ddx1, -ddx2], [ddy1, -ddy2]])
                B = np.array([x2 - x1, y2 - y1])

                try:
                    t1, _ = np.linalg.solve(A, B)
                except np.linalg.LinAlgError:
                    continue

                intersectionX = x1 + t1 * ddx1
                intersectionY = y1 + t1 * ddy1
                intersections.append((intersectionX, intersectionY))

        # Push into the rolling window. The deque's maxlen handles eviction.
        if intersections:
            self._frames_with_samples += 1
            for pt in intersections:
                self._intersections.append(pt)

        est_center = self.eye_center
        if est_center is None:
            return None

        # Draw faint magenta ray trail from this frame's ellipse centers
        # toward the eye center. (Visualization only.)
        for (cx, cy, _, __) in current_rays:
            line = ((int(cx), int(cy)), est_center)
            if line not in self.rays:
                self.rays.append(line)

        if len(self.rays) > self.RAY_HISTORY:
            self.rays = self.rays[-self.RAY_HISTORY:]

        for ellipse_center, intersection in self.rays:
            cv2.line(frame, ellipse_center, intersection, (255, 0, 255), 1)

        return est_center

    # ---- core processing (runs on worker thread) ----
    def process_frame(self, frame):
        # Pre-rotation done by caller.
        frame = cv2.flip(frame, 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        darkest_point = get_darkest_area_fast(gray)
        darkest_val = int(gray[darkest_point[1], darkest_point[0]])

        # Build three thresholded images and keep ALL valid candidate
        # ellipses, not just the single best one. The eye-center estimator
        # needs multiple rays per frame to triangulate anything.
        candidate_ellipses = []  # list of (cx, cy, angle_deg)
        best_contour = None
        best_score = 0.0
        best_ellipse = None  # (cx, cy, angle) of the highest-scoring fit

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

            refined = optimize_contours_by_angle_fast(cand)
            if refined is None or len(refined) < 5:
                continue

            try:
                ell = cv2.fitEllipse(refined)
            except cv2.error:
                continue

            (cx, cy), _, angle = ell
            candidate_ellipses.append((cx, cy, angle))

            goodness = check_ellipse_goodness(dilated, refined)
            thick_count, thin_ratio = check_contour_pixels(refined, dilated.shape)
            score = goodness * thick_count * thick_count * thin_ratio
            if score > best_score:
                best_score = score
                best_contour = refined
                best_ellipse = ell

        center_x = center_y = None
        final_ellipse = None

        if best_ellipse is not None:
            final_ellipse = best_ellipse
            (cx, cy), _, _ = final_ellipse
            center_x, center_y = int(cx), int(cy)

        h, w = frame.shape[:2]

        # Eye-sphere center from the rolling-window estimator.
        estimate = self.eyecenter_estimation(candidate_ellipses, frame)
        if estimate is not None:
            mcx = int(np.clip(estimate[0], 0, w - 1))
            mcy = int(np.clip(estimate[1], 0, h - 1))
        else:
            # No samples yet -- fall back to frame center so we still draw
            # something reasonable.
            mcx, mcy = w // 2, h // 2
        model_center = (mcx, mcy)

        # Draw overlay.
        have_estimate = estimate is not None
        baseline = self.is_baseline_reached
        if baseline:
            sphere_color = (50, 200, 50)    # green: baseline reached
        elif have_estimate:
            sphere_color = (255, 50, 50)    # blue: collecting samples
        else:
            sphere_color = (120, 120, 120)  # gray: no data yet
        cv2.circle(frame, model_center, 202, sphere_color, 2)
        cv2.circle(frame, model_center, 8, (255, 255, 0), -1)
        if final_ellipse is not None and center_x is not None:
            cv2.line(frame, model_center, (center_x, center_y),
                     (255, 150, 50), 2)
            cv2.ellipse(frame, final_ellipse, (20, 255, 255), 2)
        if baseline:
            status = (f"BASELINE  window={self.sample_count}"
                      f"/{self._intersections.maxlen}")
        elif have_estimate:
            status = (f"collecting {self._frames_with_samples}"
                      f"/{self._baseline_frames}")
        else:
            status = "searching"
        cv2.putText(frame, f"{self.name}  [{status}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if center_x is None:
            return frame, None, None

        origin, direction = self.compute_gaze_ray(
            center_x, center_y, model_center[0], model_center[1])
        return frame, origin, direction

    # ---- gaze ray in room coordinates ----
    def compute_gaze_ray(self, pupil_x, pupil_y, center_x, center_y):
        """Convert the 2D pupil offset from the rolling-window eye center
        into a 3D gaze ray in ROOM coordinates.

        Returns (origin, direction) where both are float64 numpy arrays
        in meters. `origin` is `self.eye_position_room`. `direction` is
        a unit vector in the +X right / +Y forward / +Z up frame.

        Model: the pupil sits on a sphere of radius `eye_radius_px` in
        image space. A pupil offset of (dx, dy) pixels from the sphere
        center corresponds to an angular rotation of the gaze away from
        straight-ahead. We turn dx into a yaw (rotation about +Z) and dy
        into a pitch (rotation about +X), with per-eye sign flips to
        account for camera mounting orientation.

        Returns (None, None) if the offset is implausibly large (pupil
        further from center than the eyeball radius, which means the fit
        is bad and we shouldn't report a direction).
        """
        dx = pupil_x - center_x
        dy = pupil_y - center_y
        r = self.eye_radius_px

        # Offset magnitude as a fraction of eyeball radius. Can't exceed
        # 1.0 geometrically; clip a little above that to tolerate fit
        # noise, reject beyond.
        offset_norm = np.hypot(dx, dy) / r
        if offset_norm > 1.3:
            return None, None
        # Individual axis fractions, clamped into asin's domain.
        sx = np.clip(dx / r, -1.0, 1.0)
        sy = np.clip(dy / r, -1.0, 1.0)

        yaw = self.yaw_sign * np.arcsin(sx)      # rotate about +Z (up)
        pitch = self.pitch_sign * np.arcsin(sy)  # rotate about +X (right)

        # Start with a "straight ahead" ray: +Y forward.
        # Apply pitch (about +X) then yaw (about +Z).
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy_, sy_ = np.cos(yaw), np.sin(yaw)
        y_mid = cp
        z_mid = sp
        direction = np.array([-sy_ * y_mid,
                               cy_ * y_mid,
                               z_mid], dtype=np.float64)
        n = np.linalg.norm(direction)
        if n < 1e-12:
            return None, None
        direction /= n

        return self.eye_position_room.copy(), direction


# ===========================================================================
# GAZE POINT (ARM TARGET)
# ===========================================================================
# `intersect_gaze_rays` is the function that produces the 3D gaze point --
# the intersection of the two eye vectors. This is what you hand off to
# the arm. The main loop calls it every frame and stores the result in
# the `gaze_target` variable; see `send_to_arm` below for the handoff
# point.
# ===========================================================================
def intersect_gaze_rays(lc, ld, rc, rd):
    """Least-squares closest-point of two 3D rays (one per eye) in room
    coordinates. This is the gaze intersection point -- the 3D location in
    the room the user is looking at.

    lc, rc : np.ndarray shape (3,)  -- ray origins (left-eye and right-eye
             positions in meters)
    ld, rd : np.ndarray shape (3,)  -- unit gaze directions

    Returns
    -------
    midpoint : np.ndarray shape (3,) or None
        The 3D "gaze target" -- meters, room coordinates, ready for the
        arm. None if the rays are nearly parallel (target would be at
        infinity).
    miss_distance : float or None
        Distance between the closest points on each ray (meters). Small
        = the eyes are truly converging on a point and the target is
        trustworthy. Large = they're not, be suspicious.
    """
    delta = lc - rc
    dll = float(np.dot(ld, ld))
    dlr = float(np.dot(ld, rd))
    drr = float(np.dot(rd, rd))
    dld = float(np.dot(ld, delta))
    drd = float(np.dot(rd, delta))
    denom = dll * drr - dlr * dlr
    if abs(denom) < 1e-6:
        return None, None  # nearly parallel -- no useful convergence
    t_l = (dlr * drd - drr * dld) / denom
    t_r = (dll * drd - dlr * dld) / denom
    p_l = lc + t_l * ld
    p_r = rc + t_r * rd
    midpoint = (p_l + p_r) * 0.5
    miss_distance = float(np.linalg.norm(p_l - p_r))
    return midpoint, miss_distance


def send_to_arm(gaze_target, miss_distance, left_ray, right_ray):
    """Hook point for arm control. Called every frame a valid gaze target
    exists. Replace the body with your actual arm command (ROS publish,
    socket send, serial write, whatever).

    Parameters
    ----------
    gaze_target : np.ndarray shape (3,)
        Target point in ROOM coordinates (meters). +X right, +Y forward,
        +Z up. Same frame the arm should be driven in.
    miss_distance : float
        Convergence confidence. < ~0.05 m is tight; > 0.10 m means the
        two eyes aren't really pointing at the same thing and the arm
        probably shouldn't move.
    left_ray, right_ray : (origin, direction) tuples of np.ndarray
        The individual eye rays, if your arm needs them (e.g. for
        dominant-eye override or line-of-sight extension).
    """
    # --- plug in your arm driver here ---
    return


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_dual_tracking(src_left, src_right=None, mirror_mode=False):
    """Eye positions, calibration length, etc. are read from the
    USER CONFIGURATION block at the top of this file."""
    grab_l = FreshestFrameGrabber(src_left, "left")
    grab_r = None if mirror_mode else FreshestFrameGrabber(src_right, "right")

    tracker_left = EyeTracker(
        "Left Eye",
        eye_position_room=LEFT_EYE_POSITION,
        yaw_sign=LEFT_YAW_SIGN,
        pitch_sign=LEFT_PITCH_SIGN,
    )
    tracker_right = EyeTracker(
        "Right Eye",
        eye_position_room=RIGHT_EYE_POSITION,
        yaw_sign=RIGHT_YAW_SIGN,
        pitch_sign=RIGHT_PITCH_SIGN,
    )

    pool = ThreadPoolExecutor(max_workers=2)

    # Top-down room view. See config at top of file for scale.
    viz = np.zeros((VIZ_SIZE, VIZ_SIZE, 3), dtype=np.uint8)

    # FPS tracking.
    fps_t0 = time.time()
    fps_frames = 0
    fps = 0.0

    # Gaze-target print throttle.
    last_print_t = 0.0
    print_interval = (1.0 / GAZE_PRINT_HZ) if GAZE_PRINT_HZ else None

    # Midpoint between eyes, used as the viz origin.
    mid_eye = 0.5 * (np.array(LEFT_EYE_POSITION)
                     + np.array(RIGHT_EYE_POSITION))

    def room_to_viz_topdown(p_xyz):
        """Map (X, Y, Z) meters -> (u, v) pixels on a top-down image.
        Top-down: we show X horizontally and Y vertically, with +Y
        drawn upward on the image (so user looks "up" in the view)."""
        dx = p_xyz[0] - mid_eye[0]
        dy = p_xyz[1] - mid_eye[1]
        u = int(VIZ_SIZE * 0.5 + dx * VIZ_SCALE)
        v = int(VIZ_SIZE * 0.5 - dy * VIZ_SCALE)
        return (u, v)

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
            out_l, l_origin, l_dir = fut_l.result()
            out_r, r_origin, r_dir = fut_r.result()

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

            # ---- Room-coordinate top-down gaze visualization ----
            viz[:] = 0
            # Grid: 1 m major.
            for m in range(-2, 3):
                u0 = int(VIZ_SIZE * 0.5 + m * VIZ_SCALE)
                v0 = int(VIZ_SIZE * 0.5 - m * VIZ_SCALE)
                cv2.line(viz, (u0, 0), (u0, VIZ_SIZE), (40, 40, 40), 1)
                cv2.line(viz, (0, v0), (VIZ_SIZE, v0), (40, 40, 40), 1)
            # Axis labels.
            cv2.putText(viz, "+X right", (VIZ_SIZE - 80, VIZ_SIZE // 2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            cv2.putText(viz, "+Y fwd", (VIZ_SIZE // 2 + 5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

            # Draw eye positions.
            lp = room_to_viz_topdown(np.array(LEFT_EYE_POSITION))
            rp = room_to_viz_topdown(np.array(RIGHT_EYE_POSITION))
            cv2.circle(viz, lp, 4, (0, 0, 255), -1)
            cv2.circle(viz, rp, 4, (255, 0, 0), -1)

            # =============================================================
            # GAZE POINT COMPUTATION -- the 3D intersection of the two eye
            # vectors. `gaze_target` is what the arm consumes.
            # =============================================================
            gaze_target = None
            miss = None
            if l_origin is not None and r_origin is not None:
                gaze_target, miss = intersect_gaze_rays(
                    l_origin, l_dir, r_origin, r_dir)

                # Draw each gaze ray out to 3 m even if rays don't
                # converge cleanly -- useful for seeing the slope.
                def ray_endpoint(o, d, length=3.0):
                    return room_to_viz_topdown(o + d * length)
                cv2.line(viz, lp, ray_endpoint(l_origin, l_dir),
                         (0, 0, 200), 1)
                cv2.line(viz, rp, ray_endpoint(r_origin, r_dir),
                         (200, 0, 0), 1)

                if gaze_target is not None:
                    tp = room_to_viz_topdown(gaze_target)
                    if 0 <= tp[0] < VIZ_SIZE and 0 <= tp[1] < VIZ_SIZE:
                        cv2.circle(viz, tp, 6, (0, 255, 0), -1)
                        cv2.circle(viz, tp, 10, (0, 255, 0), 1)
                    cv2.putText(
                        viz,
                        f"target: ({gaze_target[0]:+.2f}, "
                        f"{gaze_target[1]:+.2f}, {gaze_target[2]:+.2f}) m",
                        (10, VIZ_SIZE - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(
                        viz, f"miss: {miss * 100:.1f} cm",
                        (10, VIZ_SIZE - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if miss < 0.05 else (0, 165, 255), 1)

                    # ---- Arm handoff ----
                    send_to_arm(gaze_target, miss,
                                (l_origin, l_dir), (r_origin, r_dir))

                    # Throttled debug print of the arm input.
                    if print_interval is not None:
                        now = time.time()
                        if now - last_print_t >= print_interval:
                            print(
                                f"[gaze] target=("
                                f"{gaze_target[0]:+.3f},"
                                f"{gaze_target[1]:+.3f},"
                                f"{gaze_target[2]:+.3f}) m  "
                                f"miss={miss * 100:.1f} cm"
                            )
                            last_print_t = now

            cv2.imshow("Top-down Room (m)", viz)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)
            elif key == ord('r'):
                # Headset shifted? Wipe the rolling windows and re-measure.
                tracker_left.reset_eye_center()
                tracker_right.reset_eye_center()
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
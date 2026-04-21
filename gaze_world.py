# gaze_world.py
"""
Transforms per-eye gaze vectors from eye-camera frame into head frame,
then into world/room frame. Computes the 3D gaze target in real space (meters).
"""
import numpy as np

# ---------------------------------------------------------------------------
# Tunable physical constants (all meters / radians)
# ---------------------------------------------------------------------------
HUMAN_HEIGHT        = 1.70      # meters — change per subject
EYE_HEIGHT          = HUMAN_HEIGHT - 0.10   # eyes sit ~10 cm below top of head
IPD                 = 0.063     # interpupillary distance (63 mm average)
CAMERA_EYE_DISTANCE = 0.03      # 3 cm lens-to-eye
CAMERA_INWARD_ANGLE = np.radians(-35.0)  # camera rotated 35° to face eyeball

# ---------------------------------------------------------------------------
# Frames:
#   HEAD frame  — origin between the eyes, +X right, +Y up, +Z forward
#   WORLD frame — origin on the floor below the subject, +X right,
#                 +Y up, +Z forward. Floor is the plane Y=0.
# ---------------------------------------------------------------------------
LEFT_EYE_HEAD  = np.array([-IPD / 2.0, 0.0, 0.0])
RIGHT_EYE_HEAD = np.array([ IPD / 2.0, 0.0, 0.0])

# Default head pose: standing upright, looking down +Z, at the origin.
# Head origin sits at eye height above the floor.
HEAD_POSITION_WORLD = np.array([0.0, EYE_HEIGHT, 0.0])
HEAD_ROTATION_WORLD = np.eye(3)   # identity — no head yaw/pitch/roll

# ---------------------------------------------------------------------------
# Adjustable runtime parameters
# ---------------------------------------------------------------------------
GAZE_PLANE_DISTANCE = 1.0   # meters — how far in front of the subject the gaze plane sits
                            # 0.5 = arm's length, 1.0 = conversation distance,
                            # 2.0 = across-the-room, 3.0 = far wall

GAZE_GAIN_X         = 2.5   # horizontal sensitivity multiplier (side-to-side)
GAZE_GAIN_Y         = 1.0   # vertical sensitivity multiplier (up/down)
                            # The sphere model compresses angles — boost until
                            # looking fully left/right moves the target off-screen.

GAZE_SMOOTHING      = 0.85  # 0 = no smoothing (jittery), 0.95 = very smooth but laggy
                            # 0.85 is a good starting point.

FLIP_LEFT_EYE_Y     = False # set True if left eye camera is mounted upside-down
FLIP_RIGHT_EYE_Y    = False # set True if right eye camera is mounted upside-down

# Internal state for smoothing
_smoothed_target_head = None


def compute_world_gaze_projected(l_dir_cam, r_dir_cam):
    """Direction-only gaze: average both eyes' directions, project onto a
    plane at fixed distance in front of the subject. Ignores vergence.
    
    Applies X/Y gain to compensate for the sphere model's compressed angle
    output, and exponential smoothing to reduce jitter.
    """
    global _smoothed_target_head
    
    if l_dir_cam is None or r_dir_cam is None:
        return None

    l_head = transform_gaze_to_head(l_dir_cam, 'left')
    r_head = transform_gaze_to_head(r_dir_cam, 'right')
    
    # Optional per-eye Y flip (if a camera is mounted inverted)
    if FLIP_LEFT_EYE_Y:
        l_head = l_head * np.array([1.0, -1.0, 1.0])
    if FLIP_RIGHT_EYE_Y:
        r_head = r_head * np.array([1.0, -1.0, 1.0])

    # Average the two directions
    avg_dir_head = (l_head + r_head) / 2.0
    n = np.linalg.norm(avg_dir_head)
    if n < 1e-9:
        return None
    avg_dir_head /= n
    
    # Apply gain: amplify X and Y, keep Z, then re-normalize.
    # This compensates for the sphere model producing compressed angles.
    avg_dir_head = np.array([
        avg_dir_head[0] * GAZE_GAIN_X,
        avg_dir_head[1] * GAZE_GAIN_Y,
        avg_dir_head[2],
    ])
    avg_dir_head /= np.linalg.norm(avg_dir_head)

    # Origin: midpoint between eyes
    eye_mid_head = (LEFT_EYE_HEAD + RIGHT_EYE_HEAD) / 2.0

    # Project onto a plane at Z = GAZE_PLANE_DISTANCE in head frame
    if abs(avg_dir_head[2]) < 1e-6:
        target_head = eye_mid_head + avg_dir_head * GAZE_PLANE_DISTANCE
    else:
        t = (GAZE_PLANE_DISTANCE - eye_mid_head[2]) / avg_dir_head[2]
        target_head = eye_mid_head + t * avg_dir_head

    # Exponential smoothing: blend with previous target
    if _smoothed_target_head is None:
        _smoothed_target_head = target_head.copy()
    else:
        _smoothed_target_head = (
            GAZE_SMOOTHING * _smoothed_target_head
            + (1.0 - GAZE_SMOOTHING) * target_head
        )
    target_head = _smoothed_target_head.copy()

    # Head → world
    target_world     = head_to_world(target_head, is_direction=False)
    l_origin_world   = head_to_world(LEFT_EYE_HEAD,  is_direction=False)
    r_origin_world   = head_to_world(RIGHT_EYE_HEAD, is_direction=False)
    l_dir_world      = head_to_world(l_head, is_direction=True)
    r_dir_world      = head_to_world(r_head, is_direction=True)

    return {
        'left_origin_world':     l_origin_world,
        'left_direction_world':  l_dir_world,
        'right_origin_world':    r_origin_world,
        'right_direction_world': r_dir_world,
        'target_world':          target_world,
        'target_head':           target_head,
        'distance':              GAZE_PLANE_DISTANCE,
        'miss_distance':         0.0,
        'height':                target_world[1],
        'gaze_direction_head':   avg_dir_head,
    }


def reset_smoothing():
    """Call this when starting a new session to clear smoothed state."""
    global _smoothed_target_head
    _smoothed_target_head = None


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def set_head_pose(position=None, yaw=0.0, pitch=0.0, roll=0.0):
    """Update the head's pose in world frame.
    
    Args:
        position: 3-vector (x, y, z) of head origin (eye midpoint) in world.
                  If None, uses default (0, EYE_HEIGHT, 0).
        yaw:   rotation about world +Y in radians (turning left/right)
        pitch: rotation about world +X in radians (nodding up/down)
        roll:  rotation about world +Z in radians (tilting side to side)
    """
    global HEAD_POSITION_WORLD, HEAD_ROTATION_WORLD
    if position is None:
        position = np.array([0.0, EYE_HEIGHT, 0.0])
    HEAD_POSITION_WORLD = np.asarray(position, dtype=float)
    # Apply yaw, then pitch, then roll (intrinsic Y-X-Z)
    HEAD_ROTATION_WORLD = _rot_y(yaw) @ _rot_x(pitch) @ _rot_z(roll)


# ---------------------------------------------------------------------------
# Camera-to-head rotation matrices (unchanged from before)
# ---------------------------------------------------------------------------
_FLIP_Z = np.diag([1.0, 1.0, -1.0])
R_LEFT_CAM_TO_HEAD  = _rot_y(-CAMERA_INWARD_ANGLE) @ _FLIP_Z
R_RIGHT_CAM_TO_HEAD = _rot_y(+CAMERA_INWARD_ANGLE) @ _FLIP_Z


def transform_gaze_to_head(gaze_cam, eye='left'):
    gaze_cam = np.asarray(gaze_cam, dtype=float)
    R = R_LEFT_CAM_TO_HEAD if eye == 'left' else R_RIGHT_CAM_TO_HEAD
    g = R @ gaze_cam
    n = np.linalg.norm(g)
    return g / n if n > 1e-9 else g


def head_to_world(point_or_dir, is_direction=False):
    """Convert a point or direction from head frame to world frame."""
    v = np.asarray(point_or_dir, dtype=float)
    rotated = HEAD_ROTATION_WORLD @ v
    if is_direction:
        return rotated
    return rotated + HEAD_POSITION_WORLD


# ---------------------------------------------------------------------------
# Ray intersection
# ---------------------------------------------------------------------------
def closest_point_between_rays(p_a, d_a, p_b, d_b):
    p_a = np.asarray(p_a, float); p_b = np.asarray(p_b, float)
    d_a = np.asarray(d_a, float); d_b = np.asarray(d_b, float)
    na, nb = np.linalg.norm(d_a), np.linalg.norm(d_b)
    if na < 1e-9 or nb < 1e-9:
        return (p_a + p_b) / 2.0, np.inf
    d_a /= na; d_b /= nb

    w0 = p_a - p_b
    a = np.dot(d_a, d_a)
    b = np.dot(d_a, d_b)
    c = np.dot(d_b, d_b)
    d = np.dot(d_a, w0)
    e = np.dot(d_b, w0)
    denom = a * c - b * b
    if abs(denom) < 1e-9:
        return (p_a + p_b) / 2.0, np.linalg.norm(w0 - np.dot(w0, d_a) * d_a)

    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom
    pt_a = p_a + t * d_a
    pt_b = p_b + s * d_b
    return (pt_a + pt_b) / 2.0, np.linalg.norm(pt_a - pt_b)


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------
def compute_world_gaze(l_dir_cam, r_dir_cam):
    """Full pipeline: camera-frame gazes → world-frame target point.

    Returns dict with:
        left_origin_world,  left_direction_world   — left ray in world frame
        right_origin_world, right_direction_world  — right ray in world frame
        target_world       — 3D gaze target (meters, floor-origin world frame)
        target_head        — same target expressed in head frame
        distance           — distance from eye midpoint to target
        miss_distance      — how close the two rays came (stereo confidence)
        height             — Y coordinate of target (useful: floor=0, eye=EYE_HEIGHT)
    """
    if l_dir_cam is None or r_dir_cam is None:
        return None

    # Camera → head
    l_dir_head = transform_gaze_to_head(l_dir_cam, 'left')
    r_dir_head = transform_gaze_to_head(r_dir_cam, 'right')

    # Intersect in head frame first (cheaper, same result after rigid transform)
    target_head, miss = closest_point_between_rays(
        LEFT_EYE_HEAD,  l_dir_head,
        RIGHT_EYE_HEAD, r_dir_head,
    )

    # Head → world
    target_world       = head_to_world(target_head, is_direction=False)
    l_origin_world     = head_to_world(LEFT_EYE_HEAD,  is_direction=False)
    r_origin_world     = head_to_world(RIGHT_EYE_HEAD, is_direction=False)
    l_dir_world        = head_to_world(l_dir_head, is_direction=True)
    r_dir_world        = head_to_world(r_dir_head, is_direction=True)

    eye_mid_world = (l_origin_world + r_origin_world) / 2.0
    distance = np.linalg.norm(target_world - eye_mid_world)

    return {
        'left_origin_world':     l_origin_world,
        'left_direction_world':  l_dir_world,
        'right_origin_world':    r_origin_world,
        'right_direction_world': r_dir_world,
        'target_world':          target_world,
        'target_head':           target_head,
        'distance':              distance,
        'miss_distance':         miss,
        'height':                target_world[1],
    }

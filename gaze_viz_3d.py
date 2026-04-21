import matplotlib
# Qt5Agg: avoids "Cannot load backend 'TkAgg' ... as 'qt' is currently running"
# when OpenCV/Qt loads first. Requires PyQt5 (see project requirements).
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Physical scene constants (all in meters)
# ---------------------------------------------------------------------------
# Origin = midpoint between the two eyes.
# +X = right (from the person's perspective looking forward)
# +Y = up
# +Z = forward (the direction the person is looking)
#
# Typical adult interpupillary distance is ~0.063 m (63 mm).
# Tweak these as needed for your subject / robot-arm coordinate frame.

IPD = 0.063  # interpupillary distance
LEFT_EYE_POS  = np.array([-IPD / 2, 1.0, 1.0])
RIGHT_EYE_POS = np.array([ IPD / 2, 1.0, 1.0])

# Scene extents (meters). A roughly 2 m cube in front of the eyes.
SCENE_XLIM = (-1.0, 1.0)
SCENE_YLIM = (-1.0, 1.0)
SCENE_ZLIM = (-0.2, 2.0)   # small negative z so the eyes at z=0 are visible

# Fallback ray length when no valid intersection exists (meters)
_FALLBACK_RAY_LEN = 1.0

_fig = None
_ax = None
_vector_artists = []   # quiver + scatter artists we redraw every frame


def start():
    """Set up the 3D figure once.  Draws fixed eye markers; vectors are
    drawn in update()."""
    global _fig, _ax
    plt.ion()
    _fig = plt.figure(1, figsize=(6, 5))
    _ax = _fig.add_subplot(111, projection='3d')
    _ax.set_title("3D Gaze Vectors (meters)")

    _ax.set_xlabel('X (m)')
    _ax.set_ylabel('Y (m)')
    _ax.set_zlabel('Z (m) — forward')
    _ax.set_xlim(*SCENE_XLIM)
    _ax.set_ylim(*SCENE_YLIM)
    _ax.set_zlim(*SCENE_ZLIM)

    # Fixed eye markers — drawn once and never cleared
    _ax.scatter(*LEFT_EYE_POS,  color='red',  s=80, label='Left Eye')
    _ax.scatter(*RIGHT_EYE_POS, color='blue', s=80, label='Right Eye')
    _ax.legend(loc='upper left')

    plt.show(block=False)


def update(l_direction, r_direction, intersection_3d=None):
    """Redraw only the two gaze vectors and the intersection marker.
    The eyes stay put.

    Args:
        l_direction: 3D unit vector, left eye's gaze direction (in eye-frame)
        r_direction: same, for the right eye
        intersection_3d: optional 3D point where the rays converge
    """
    global _fig, _ax, _vector_artists
    if _ax is None or _fig is None:
        return

    # Remove only the per-frame artists we added last time
    for artist in _vector_artists:
        try:
            artist.remove()
        except (ValueError, NotImplementedError):
            pass
    _vector_artists = []

    l_dir = np.asarray(l_direction, dtype=float)
    r_dir = np.asarray(r_direction, dtype=float)

    if intersection_3d is not None:
        intersection_3d = np.asarray(intersection_3d, dtype=float)
        a_vec = intersection_3d - LEFT_EYE_POS
        b_vec = intersection_3d - RIGHT_EYE_POS

        q_l = _ax.quiver(*LEFT_EYE_POS,  *a_vec, color='red',  linewidth=2)
        q_r = _ax.quiver(*RIGHT_EYE_POS, *b_vec, color='blue', linewidth=2)
        pt  = _ax.scatter(*intersection_3d, color='green', s=60)
        _vector_artists.extend([q_l, q_r, pt])
    else:
        q_l = _ax.quiver(*LEFT_EYE_POS,  *l_dir,
                         color='red',  linewidth=2, length=_FALLBACK_RAY_LEN)
        q_r = _ax.quiver(*RIGHT_EYE_POS, *r_dir,
                         color='blue', linewidth=2, length=_FALLBACK_RAY_LEN)
        _vector_artists.extend([q_l, q_r])

    _fig.canvas.draw_idle()
    _fig.canvas.flush_events()


def stop():
    plt.ioff()
    if _fig is not None:
        plt.close(_fig)


def compute_gaze_intersection(p_a, d_a, p_b, d_b):
    """Closest point between two 3D rays (midpoint of the shortest
    perpendicular segment between them).  All arguments must be in the
    same coordinate frame (meters)."""
    d_a = np.asarray(d_a, dtype=float)
    d_b = np.asarray(d_b, dtype=float)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)

    norm_a = np.linalg.norm(d_a)
    norm_b = np.linalg.norm(d_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return (p_a + p_b) / 2.0
    d_a = d_a / norm_a
    d_b = d_b / norm_b

    w0 = p_a - p_b
    a = np.dot(d_a, d_a)
    b = np.dot(d_a, d_b)
    c = np.dot(d_b, d_b)
    d = np.dot(d_a, w0)
    e = np.dot(d_b, w0)
    denom = a * c - b * b

    if abs(denom) < 1e-9:
        return (p_a + p_b) / 2.0  # parallel

    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom
    point_a = p_a + t * d_a
    point_b = p_b + s * d_b
    return (point_a + point_b) / 2.0
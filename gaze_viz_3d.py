import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import numpy as np

from gaze_world import (
    EYE_HEIGHT, IPD, HEAD_POSITION_WORLD, head_to_world, LEFT_EYE_HEAD, RIGHT_EYE_HEAD,
)

# World-frame scene extents (meters)
SCENE_XLIM = (-1.5, 1.5)
SCENE_YLIM = (0.0,  2.2)    # floor to just above head
SCENE_ZLIM = (-0.3, 3.0)

_FALLBACK_RAY_LEN = 1.0
_fig = None
_ax = None
_vector_artists = []


def start():
    global _fig, _ax
    plt.ion()
    _fig = plt.figure(1, figsize=(7, 6))
    _ax = _fig.add_subplot(111, projection='3d')
    _ax.set_title("3D Gaze in World Frame (meters)")
    _ax.set_xlabel('X (m) — right')
    _ax.set_ylabel('Z (m) — forward')   # swap so forward reads left-to-right
    _ax.set_zlabel('Y (m) — up')
    _ax.set_xlim(*SCENE_XLIM)
    _ax.set_ylim(*SCENE_ZLIM)
    _ax.set_zlim(*SCENE_YLIM)

    # Floor plane at Y=0
    xs = np.linspace(SCENE_XLIM[0], SCENE_XLIM[1], 2)
    zs = np.linspace(SCENE_ZLIM[0], SCENE_ZLIM[1], 2)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros_like(X)
    _ax.plot_surface(X, Z, Y, alpha=0.15, color='gray')

    # Eyes in world frame
    le = head_to_world(LEFT_EYE_HEAD)
    re = head_to_world(RIGHT_EYE_HEAD)
    _ax.scatter(le[0], le[2], le[1], color='red',  s=80, label='Left Eye')
    _ax.scatter(re[0], re[2], re[1], color='blue', s=80, label='Right Eye')

    # Vertical line showing body/height
    _ax.plot([0, 0], [0, 0], [0, EYE_HEIGHT], color='black', linewidth=1, alpha=0.4)
    _ax.legend(loc='upper left')
    plt.show(block=False)


def update(l_origin, l_dir, r_origin, r_dir, target=None):
    """All args are in WORLD frame (meters)."""
    global _vector_artists
    if _ax is None:
        return

    for a in _vector_artists:
        try: a.remove()
        except (ValueError, NotImplementedError): pass
    _vector_artists = []

    # Note axis swap: matplotlib (X, Z, Y) order because we plotted Y as vertical
    def _plot_xyz(v):
        return v[0], v[2], v[1]

    if target is not None:
        target = np.asarray(target, float)
        lvec = target - l_origin
        rvec = target - r_origin
        q_l = _ax.quiver(*_plot_xyz(l_origin), lvec[0], lvec[2], lvec[1],
                         color='red', linewidth=2)
        q_r = _ax.quiver(*_plot_xyz(r_origin), rvec[0], rvec[2], rvec[1],
                         color='blue', linewidth=2)
        pt  = _ax.scatter(target[0], target[2], target[1],
                          color='green', s=70)
        _vector_artists.extend([q_l, q_r, pt])
    else:
        q_l = _ax.quiver(*_plot_xyz(l_origin), l_dir[0], l_dir[2], l_dir[1],
                         color='red', linewidth=2, length=_FALLBACK_RAY_LEN)
        q_r = _ax.quiver(*_plot_xyz(r_origin), r_dir[0], r_dir[2], r_dir[1],
                         color='blue', linewidth=2, length=_FALLBACK_RAY_LEN)
        _vector_artists.extend([q_l, q_r])

    _fig.canvas.draw_idle()
    _fig.canvas.flush_events()


def stop():
    plt.ioff()
    if _fig is not None:
        plt.close(_fig)
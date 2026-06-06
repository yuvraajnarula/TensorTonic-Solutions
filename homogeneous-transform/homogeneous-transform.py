import numpy as np

def apply_homogeneous_transform(T, points):
    points = np.array(points)
    T = np.array(T)
    is_single = points.ndim == 1
    if is_single:
        points = points[np.newaxis, :]

    ones = np.ones((points.shape[0], 1))
    _ = np.hstack([points, ones ])
    _h = _ @ T.T
    _3d = _h[:,:3]
    return _3d[0] if is_single else _3d
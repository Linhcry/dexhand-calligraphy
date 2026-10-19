"""SI units; T_base_camera maps OpenCV optical (+x right,+y down,+z forward) to base."""
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass(frozen=True)
class Calibration:
    K: np.ndarray
    T_base_camera: np.ndarray
    image_size: tuple
    version: str
    distortion: tuple = (0., 0., 0., 0., 0.)
    base_frame: str = 'robot_base'

    def __post_init__(self):
        K, T = np.asarray(self.K, float), np.asarray(self.T_base_camera, float)
        if K.shape != (3, 3) or T.shape != (4, 4) or not np.isfinite(K).all() or not np.isfinite(T).all():
            raise ValueError('Invalid camera matrices')
        if not np.allclose(T[3], [0, 0, 0, 1]) or not np.allclose(T[:3,:3].T @ T[:3,:3], np.eye(3), atol=1e-6) or np.linalg.det(T[:3,:3]) < .999:
            raise ValueError('Extrinsics must be a rigid transform in metres')
        if K[0,0] <= 0 or K[1,1] <= 0 or len(self.image_size) != 2 or min(self.image_size) <= 0 or not self.version or not self.base_frame:
            raise ValueError('Missing calibration metadata')
        object.__setattr__(self, 'K', K)
        object.__setattr__(self, 'T_base_camera', T)

    @classmethod
    def from_dict(cls, d):
        if d.get('units') != 'm' or d.get('optical_convention') != 'opencv':
            raise ValueError('Calibration must declare metres and OpenCV optical axes')
        return cls(d['K'], d['T_base_camera'], tuple(d['image_size']), d['version'], tuple(d.get('distortion', (0,)*5)),d['base_frame'])

    def to_dict(self):
        return dict(K=self.K.tolist(), T_base_camera=self.T_base_camera.tolist(), image_size=self.image_size,
                    version=self.version, distortion=self.distortion, units='m', optical_convention='opencv',base_frame=self.base_frame)

    def rays(self, pixels):
        xy = cv2.undistortPoints(np.asarray(pixels, float).reshape(-1,1,2), self.K, np.asarray(self.distortion, float)).reshape(-1,2)
        return np.c_[xy, np.ones(len(xy))] @ self.T_base_camera[:3,:3].T

    def backproject(self, pixels, depth_m):
        z = np.asarray(depth_m, float).reshape(-1)
        if not np.isfinite(z).all() or (z <= 0).any():
            raise ValueError('Depth must be finite positive optical Z in metres, aligned to RGB')
        return self.T_base_camera[:3,3] + self.rays(pixels) * z[:,None]

    def intersect_plane(self, pixels, plane):
        p = np.asarray(plane, float); rays = self.rays(pixels); origin = self.T_base_camera[:3,3]
        denominator = rays @ p[:3]
        if (np.abs(denominator) < 1e-8).any():
            raise ValueError('Ray parallel to plane')
        scale = -(origin @ p[:3] + p[3]) / denominator
        if (scale <= 0).any(): raise ValueError('Plane behind camera')
        return origin + scale[:,None]*rays


def fit_table(points, threshold=.002, seed=17):
    """RANSAC horizontal dominant plane; return normal up, plane offset and residual."""
    p = np.asarray(points, float)
    p = p[np.isfinite(p).all(1)]
    if len(p) < 100: raise ValueError('Insufficient depth for table')
    rng = np.random.default_rng(seed); best = np.zeros(len(p), bool)
    for _ in range(80):
        a,b,c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b-a,c-a); length = np.linalg.norm(n)
        if length < 1e-9: continue
        n /= length
        if abs(n[2]) < .95: continue
        mask = np.abs((p-a)@n) < threshold
        if mask.sum() > best.sum(): best = mask
    if best.mean() < .25: raise ValueError('No dominant table plane')
    mean = p[best].mean(0); _,_,v = np.linalg.svd(p[best]-mean, full_matrices=False)
    n = v[-1] * (1 if v[-1,2] >= 0 else -1)
    plane = np.r_[n,-mean@n]
    return plane, float(np.sqrt(np.mean((p[best]@n+plane[3])**2)))

"""Local image-based perception; callers must provide measured calibration points."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class PlaneCalibration:
    homography: np.ndarray

    @classmethod
    def from_correspondences(cls, pixels, plane_xy):
        pixels, plane_xy = np.asarray(pixels, float), np.asarray(plane_xy, float)
        if pixels.shape != plane_xy.shape or pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < 4:
            raise ValueError("Need at least four corresponding image and metric plane points")
        if not np.isfinite(pixels).all() or not np.isfinite(plane_xy).all():
            raise ValueError("Calibration points must be finite")
        if np.linalg.matrix_rank(pixels - pixels.mean(0)) < 2 or np.linalg.matrix_rank(plane_xy - plane_xy.mean(0)) < 2:
            raise ValueError("Collinear calibration points cannot define a plane mapping")
        # Inputs are already measured correspondences, not unverified feature matches.
        h, _ = cv2.findHomography(pixels, plane_xy, method=0)
        if h is None or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-15:
            raise ValueError("Degenerate calibration")
        return cls(h)

    def to_plane(self, pixels):
        p = np.asarray(pixels, dtype=float).reshape(-1, 2)
        h = np.c_[p, np.ones(len(p))] @ self.homography.T
        if (np.abs(h[:, 2]) < 1e-12).any():
            raise ValueError("Point maps to the plane horizon")
        return h[:, :2] / h[:, 2:3]


def circle_candidates(bgr, min_radius=3, max_radius=80):
    """Circular image features only; not a trained dragon-eye detector."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cv2.GaussianBlur(gray, (5, 5), 1), cv2.HOUGH_GRADIENT,
                              dp=1, minDist=12, param1=100, param2=20,
                              minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        return []
    return [{"center_px": [float(x), float(y)], "radius_px": float(r),
             "semantic_eye_verified": False} for x, y, r in circles[0]]


class OfflineOCR:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        from .paths import ASSETS
        folder=ASSETS/'ocr'
        models={}
        if folder.exists():
            for key,name in [('det_model_path','ch_PP-OCRv4_det_infer.onnx'),('cls_model_path','ch_ppocr_mobile_v2.0_cls_infer.onnx'),('rec_model_path','ch_PP-OCRv4_rec_infer.onnx')]:
                path=folder/name
                if not path.is_file():raise FileNotFoundError(f'Incomplete bundled OCR models: {path}')
                models[key]=str(path)
        self.engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1,
                               det_use_cuda=False, cls_use_cuda=False, rec_use_cuda=False,**models)

    def read(self, bgr):
        results, _ = self.engine(bgr)
        return [{"box": np.asarray(box).tolist(), "text": text, "confidence": float(score)}
                for box, text, score in (results or [])]

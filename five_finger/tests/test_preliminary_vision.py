from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from experiments.preliminary_dragon_eye import detect_eye, detect_paper


def test_preliminary_camera_image_paper_and_eye_localization() -> None:
    paper_center = [0.20, 0.0]
    paper_size = [0.297, 0.420]
    eye_local = [0.04, -0.02]
    eye_radius_m = 0.008
    corners = np.array([[552, 375], [1012, 375], [1012, 1080], [552, 1080]], float)

    image = Image.new("RGB", (1600, 1200), (76, 78, 84))
    draw = ImageDraw.Draw(image)
    draw.polygon([tuple(p) for p in corners], fill=(247, 242, 230))
    w, h = 1500, 2100
    eye_px = (
        int((eye_local[0] / paper_size[0] + 0.5) * w),
        int((0.5 - eye_local[1] / paper_size[1]) * h),
    )
    eye_r_px = max(6, int(round(eye_radius_m / paper_size[0] * w)))
    # Warp a paper-sized image containing the eye into the camera quadrilateral.
    paper_image = Image.new("RGB", (w, h), (247, 242, 230))
    ImageDraw.Draw(paper_image).ellipse(
        (eye_px[0] - eye_r_px, eye_px[1] - eye_r_px, eye_px[0] + eye_r_px, eye_px[1] + eye_r_px),
        fill=(24, 28, 34),
    )
    transform = cv2.getPerspectiveTransform(
        np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32),
        corners.astype(np.float32),
    )
    warped = cv2.warpPerspective(np.asarray(paper_image)[:, :, ::-1], transform, (1600, 1200))
    mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 8
    background = np.full((1200, 1600, 3), (76, 78, 84), np.uint8)
    background[mask] = warped[mask]
    rgb = background[:, :, ::-1]

    paper = detect_paper(rgb, {"paper_size_m": paper_size, "paper_center_xy_m": paper_center})
    eye = detect_eye(rgb, paper)
    assert abs(eye["center_base_m"][0] - (paper_center[0] + eye_local[0])) < 0.002
    assert abs(eye["center_base_m"][1] - (paper_center[1] + eye_local[1])) < 0.002
    assert abs(eye["radius_m"] - eye_radius_m) < 0.0015

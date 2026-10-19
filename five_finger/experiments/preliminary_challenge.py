"""Independent 2026 preliminary examiner: draw a practice dragon and score the dot.

The controller may read ``public/`` pixels and published scene geometry only.
``examiner_private/`` contains the true eye centre/radius and is protected by an
audit hook in the controller. The image is a development proxy, not the official
dragon-image library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from fivefinger.competition_rules import (
    load_competition_rules,
    operation_mode_multiplier,
    score_preliminary_attempt,
)
from fivefinger.paths import ROOT

CASES = ROOT / "outputs" / "preliminary_2026"
PRE_SETTINGS = ROOT / "config" / "competition_scene_2026_preliminary.json"


def _paper_size():
    return np.asarray(json.loads(PRE_SETTINGS.read_text(encoding="utf-8"))["paper_size_m"], float)


def _make_manifest():
    files = list((ROOT / "fivefinger").rglob("*.py"))
    files += [
        ROOT / "experiments/preliminary_challenge.py",
        ROOT / "experiments/preliminary_dragon_eye.py",
        ROOT / "experiments/run_preliminary_round.py",
        ROOT / "experiments/competition_scene.py",
        ROOT / "experiments/writing.py",
        ROOT / "experiments/probe_grasp.py",
        ROOT / "experiments/blind_competition.py",
        ROOT / "experiments/writing_config.json",
        ROOT / "config/competition_scene_2026_preliminary.json",
        ROOT / "config/competition_20260911.json",
        ROOT / "config/competition_scene.json",
        ROOT / "config/model_sources.json",
    ]
    files += [p for p in (ROOT / "config").glob("*.json")]
    unique = sorted({p.resolve() for p in files if p.is_file()})
    return {
        str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in unique
    }


def _draw_dragon(image: Image.Image, eye_px: tuple[int, int], eye_r_px: int) -> None:
    draw = ImageDraw.Draw(image)
    w, h = image.size
    ink = (68, 64, 62)
    pale = (124, 116, 110)
    # Stylised non-official dragon line art. Thin curves deliberately avoid
    # creating another large filled circle that a simple eye detector could
    # confuse with the target eye.
    for y in range(0, h, 3):
        x = int(w / 2 + w * 0.31 * np.sin(7.0 * y / h) * np.sin(3.0 * y / h))
        r = 3 + int(2 * y / h)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=ink, width=4)
    draw.arc((int(w * 0.18), int(h * 0.06), int(w * 0.82), int(h * 0.94)),
             start=10, end=250, fill=pale, width=5)
    draw.arc((int(w * 0.30), int(h * 0.12), int(w * 0.70), int(h * 0.88)),
             start=200, end=70, fill=pale, width=4)
    # Head outline: two strokes near the target, but no closed dark disk.
    cx, cy = eye_px
    draw.line((cx - eye_r_px * 2, cy + eye_r_px, cx - eye_r_px, cy + eye_r_px * 2), fill=ink, width=5)
    draw.line((cx + eye_r_px, cy + eye_r_px * 2, cx + eye_r_px * 2, cy + eye_r_px), fill=ink, width=5)
    # Target eye: the only filled circular dark region on the paper.
    draw.ellipse((cx - eye_r_px, cy - eye_r_px, cx + eye_r_px, cy + eye_r_px), fill=(24, 28, 34))


def create(development_eye: dict | None = None):
    settings = json.loads(PRE_SETTINGS.read_text(encoding="utf-8"))
    paper = settings["paper_center_xy_m"]
    size = _paper_size()
    if development_eye is None:
        eye_r_m = float(np.round(secrets.SystemRandom().uniform(0.008, 0.011), 4))
        margin = 0.045 + eye_r_m
        eye_local_m = np.array([
            float(np.round(secrets.SystemRandom().uniform(-(size[0] / 2 - margin), size[0] / 2 - margin), 4)),
            float(np.round(secrets.SystemRandom().uniform(-(size[1] / 2 - margin), size[1] / 2 - margin), 4)),
        ])
    else:
        eye_r_m = float(development_eye["radius_m"])
        eye_local_m = np.asarray(development_eye["center_local_m"], float)
        if eye_r_m <= 0 or eye_local_m.shape != (2,):
            raise ValueError("Invalid development eye definition")

    eye_base_m = np.array(paper) + eye_local_m
    w, h = 1500, 2100
    eye_px = (
        int(round((eye_local_m[0] / size[0] + 0.5) * w)),
        int(round((0.5 - eye_local_m[1] / size[1]) * h)),
    )
    eye_r_px = max(6, int(round(eye_r_m / size[0] * w)))
    paper_image = Image.new("RGB", (w, h), (247, 242, 230))
    _draw_dragon(paper_image, eye_px, eye_r_px)

    # The examiner renders the practice dragon into a fixed overhead-camera view.
    # This keeps the controller's vision input an image file while the physical
    # MuJoCo scene stays a clean paper for grasp/dip/dot dynamics.
    cam_w, cam_h = settings["camera_resolution"]
    cam_corners = np.asarray(settings["camera_paper_corners_px"], dtype=np.float32)
    src_corners = np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(src_corners, cam_corners)
    warped = cv2.warpPerspective(
        np.asarray(paper_image)[:, :, ::-1], transform, (cam_w, cam_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 8
    background = np.full((cam_h, cam_w, 3), (76, 78, 84), dtype=np.uint8)
    background[warped_mask] = warped[warped_mask]
    camera_image = Image.fromarray(background[:, :, ::-1])

    nonce = secrets.token_hex(32)
    case = CASES / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    public = case / "public"
    private = case / "examiner_private"
    public.mkdir(parents=True)
    private.mkdir()
    camera_image.save(public / "dragon_pixels.png")
    answer = {
        "eye_center_local_m": eye_local_m.tolist(),
        "eye_center_base_m": eye_base_m.tolist(),
        "eye_radius_m": eye_r_m,
        "nonce": nonce,
    }
    public_metadata = {
        "case_id": case.name,
        "evaluation_mode": "development_known_to_evaluator" if development_eye else "random_blind_draw",
        "paper_size_m": size.tolist(),
        "paper_center_xy_m": paper,
        "paper_top_z_m": settings["paper_top_z_m"],
        "camera_resolution": settings["camera_resolution"],
        "camera_fovy_deg": settings["camera_fovy_deg"],
        "camera_paper_corners_px": settings["camera_paper_corners_px"],
        "vision_input_mode": "examiner_rendered_overhead_camera_image",
        "dragon_image_sha256": hashlib.sha256((public / "dragon_pixels.png").read_bytes()).hexdigest(),
        "official_dragon_pattern": False,
        "answer_commitment_sha256": hashlib.sha256(
            (nonce + json.dumps(answer["eye_center_base_m"]) + json.dumps(answer["eye_radius_m"])).encode()
        ).hexdigest(),
        "examiner_note": "Non-official practice dragon; controller receives image bytes and published paper/camera geometry only.",
    }
    (public / "scene.json").write_text(json.dumps(public_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (public / "controller_manifest.json").write_text(
        json.dumps(_make_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (private / "answer.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
    CASES.mkdir(exist_ok=True)
    (CASES / "latest_case.txt").write_text(case.name, encoding="utf-8")
    # Never print the true eye location.
    print(json.dumps({"case_id": case.name, "answer_commitment_sha256": public_metadata["answer_commitment_sha256"]}, ensure_ascii=False), flush=True)
    return case


def judge(case):
    folder = CASES / case
    public = json.loads((folder / "public/scene.json").read_text(encoding="utf-8"))
    secret = json.loads((folder / "examiner_private/answer.json").read_text(encoding="utf-8"))
    report_path = folder / "run/report.json"
    if not report_path.exists():
        verdict = {
            "score": None,
            "status": "controller_failed_before_report",
            "commitment_valid": False,
        }
        (folder / "judge.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(verdict, ensure_ascii=False), flush=True)
        return verdict

    report = json.loads(report_path.read_text(encoding="utf-8"))
    committed = hashlib.sha256(
        (secret["nonce"] + json.dumps(secret["eye_center_base_m"]) + json.dumps(secret["eye_radius_m"])).encode()
    ).hexdigest() == public["answer_commitment_sha256"]
    dot = report.get("dot", {})
    dot_center = np.asarray(dot.get("center_base_m", [np.nan, np.nan]), float)
    dot_radius = float(dot.get("radius_m", 0.0))
    center_error = (
        float(np.linalg.norm(dot_center[:2] - np.asarray(secret["eye_center_base_m"])[:2]))
        if np.isfinite(dot_center).all()
        else float("inf")
    )
    ink_outer_radius_m = center_error + dot_radius if np.isfinite(center_error) else float("inf")
    rules = load_competition_rules()
    score = score_preliminary_attempt(
        touched_brush=bool(report.get("process_evidence", {}).get("touched_brush", False)),
        stable_hold_3s=bool(report.get("process_evidence", {}).get("stable_hold_3s", False)),
        dipped_ink=bool(report.get("process_evidence", {}).get("dipped_ink", False)),
        ink_outer_radius_m=ink_outer_radius_m,
        ink_radius_m=dot_radius,
        eye_radius_m=float(secret["eye_radius_m"]),
        penalty_points=float(report.get("penalty_points", 0.0)),
        operation_mode_multiplier=operation_mode_multiplier(rules),
    )
    verdict = {
        "commitment_valid": committed,
        "status": report.get("stop_reason"),
        "elapsed_s": report.get("elapsed_sim_s"),
        "elapsed_within_180s": bool(0.0 <= float(report.get("elapsed_sim_s", 180.1)) <= 180.0),
        "true_eye_radius_m": secret["eye_radius_m"],
        "true_eye_center_base_m": secret["eye_center_base_m"],
        "reported_dot_center_base_m": dot.get("center_base_m"),
        "reported_dot_radius_m": dot_radius,
        "dot_center_error_mm": center_error * 1000,
        "ink_outer_radius_mm": ink_outer_radius_m * 1000,
        "official_score_pending_human_verification": True,
        **score,
    }
    (folder / "judge.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False), flush=True)
    return verdict


def main():
    p = argparse.ArgumentParser(description="2026 preliminary dragon-eye examiner")
    p.add_argument("action", choices=["create", "judge"])
    p.add_argument("--case")
    p.add_argument("--development-eye-radius", type=float)
    p.add_argument("--development-eye-x", type=float)
    p.add_argument("--development-eye-y", type=float)
    a = p.parse_args()
    if a.action == "create":
        dev = None
        if a.development_eye_radius is not None:
            dev = {
                "radius_m": a.development_eye_radius,
                "center_local_m": [a.development_eye_x or 0.0, a.development_eye_y or 0.0],
            }
        create(dev)
    else:
        judge(a.case or (CASES / "latest_case.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

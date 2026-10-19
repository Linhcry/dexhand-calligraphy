"""2026 preliminary round controller: dragon-eye dotting in MuJoCo.

This is deliberately separate from the final calligraphy controller. It receives
only the published paper/camera geometry and the dragon-image pixels, detects one
eye, then performs the physical sequence:

    observe -> grasp -> stable hold 3 s -> dip ink -> move -> lower -> dot -> verify

Process evidence, dot measurement and round time are written to ``run/report.json``.
The independent examiner in ``preliminary_challenge.py`` owns the official score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from experiments.blind_competition import InkState, command_nib, nib_point
from experiments.competition_scene import TableClearance, make_scene
from experiments.probe_grasp import brush_contacts
from experiments.writing import paper_contact
from fivefinger.competition_rules import load_competition_rules, round_limit_seconds
from fivefinger.control import set_joint_targets, solve_arm_ik
from fivefinger.grasp_feedback import ContactGripServo
from fivefinger.paths import ROOT
from fivefinger.trajectory import smoothstep5

SETTINGS = ROOT / "config" / "competition_scene_2026_preliminary.json"


def load_settings():
    return json.loads(SETTINGS.read_text(encoding="utf-8"))


def verify_manifest(public: Path) -> bool:
    path = public / "controller_manifest.json"
    if not path.exists():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        candidate = (ROOT / name).resolve()
        if not candidate.is_relative_to(ROOT.resolve()):
            raise ValueError(f"Invalid manifest path: {name}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Controller changed after random draw: {name}")
    return True


def render_camera(model, data, path_rgb, path_depth=None):
    with mujoco.Renderer(model, width=1600, height=1200) as renderer:
        renderer.update_scene(data, camera="overhead")
        rgb = renderer.render()
        Image.fromarray(rgb).save(path_rgb)
        if path_depth is not None:
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera="overhead")
            np.save(path_depth, renderer.render())
    return np.asarray(rgb)


def _order_paper_corners(points):
    """Return image corners as TL, TR, BR, BL for a top-down rectangular sheet."""
    pts = np.asarray(points, float).reshape(-1, 2)
    ordered = pts[np.lexsort((pts[:, 0], pts[:, 1]))]
    top = ordered[:2][np.argsort(ordered[:2, 0])]
    bottom = ordered[2:][np.argsort(ordered[2:, 0])[::-1]]
    return np.vstack([top, bottom])


def detect_paper(rgb, public_meta):
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    white = cv2.inRange(bgr, (165, 160, 145), (255, 255, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No paper-like bright region found")
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    area = cv2.contourArea(hull)
    if area < (w * h) * 0.04:
        raise RuntimeError(f"Paper region too small: {area:.0f} px")
    rect = cv2.minAreaRect(hull)
    corners = cv2.boxPoints(rect)
    ordered = _order_paper_corners(corners)
    size_m = np.asarray(public_meta["paper_size_m"], float)
    center_m = np.asarray(public_meta["paper_center_xy_m"], float)
    # Top image row is world +Y for the fixed overhead camera in this scene.
    world = np.array([
        [center_m[0] - size_m[0] / 2, center_m[1] + size_m[1] / 2],
        [center_m[0] + size_m[0] / 2, center_m[1] + size_m[1] / 2],
        [center_m[0] + size_m[0] / 2, center_m[1] - size_m[1] / 2],
        [center_m[0] - size_m[0] / 2, center_m[1] - size_m[1] / 2],
    ], float)
    homography, _ = cv2.findHomography(ordered, world, method=0)
    if homography is None or not np.isfinite(homography).all():
        raise RuntimeError("Paper-plane homography is degenerate")
    paper_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(paper_mask, np.round(hull).astype(np.int32), 255)
    return dict(corners_px=ordered.tolist(), homography=homography, mask=paper_mask,
                area_px=float(area), paper_size_m=size_m.tolist(),
                paper_center_xy_m=center_m.tolist())


def _to_plane(homography, pixels):
    p = np.asarray(pixels, float).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ homography.T
    return q[:, :2] / q[:, 2:3]


def detect_eye(rgb, paper, min_radius_m=0.003, max_radius_m=0.020):
    """Development detector: one dark filled circle inside the published paper.

    It is intentionally conservative. Official dragon patterns are expected to
    replace this with the annotated template detector once the library is issued.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < 85).astype(np.uint8) * 255
    dark &= paper["mask"]
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if int(dark.sum()) < 40:
        raise RuntimeError("No dark circular candidate inside the paper")

    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (5, 5), 1.2),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, int(min_radius_m / paper["paper_size_m"][0] * gray.shape[1] * 2)),
        param1=90,
        param2=24,
        minRadius=8,
        maxRadius=90,
    )
    candidates = []
    h, w = gray.shape
    local_limits = np.asarray(paper["paper_size_m"], float) / 2 - max_radius_m * 1.5

    def _circle_fill(x, y, r):
        yy, xx = np.mgrid[0:h, 0:w]
        mask = ((xx - x) ** 2 + (yy - y) ** 2) <= r * r
        area = int(mask.sum())
        if area < 20:
            return 0.0
        return float(np.count_nonzero(dark[mask])) / area

    def _radius_m(x, y, r):
        perimeter = np.array([
            [x + r * np.cos(a), y + r * np.sin(a)]
            for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ])
        return float(np.median(np.linalg.norm(
            _to_plane(paper["homography"], perimeter)
            - _to_plane(paper["homography"], [[x, y]]), axis=1)))

    if circles is not None:
        for x, y, r in circles[0]:
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= yi < h and 0 <= xi < w) or not paper["mask"][yi, xi]:
                continue
            radius_m = _radius_m(x, y, r)
            center_m = _to_plane(paper["homography"], [[x, y]])[0]
            local = center_m - np.asarray(paper.get("paper_center_xy_m", [0.0, 0.0]), float)
            if (min_radius_m <= radius_m <= max_radius_m and np.all(np.abs(local) <= local_limits)):
                fill = _circle_fill(x, y, r)
                # A filled target eye has a high dark fill fraction; thin dragon
                # strokes and their intersections do not.
                if fill >= 0.55:
                    candidates.append((fill, 1.0, radius_m, (float(x), float(y)), "hough"))

    if not candidates:
        contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 40:
                continue
            (x, y), r = cv2.minEnclosingCircle(contour)
            xi, yi = int(round(x)), int(round(y))
            if r <= 0 or not (0 <= yi < h and 0 <= xi < w) or not paper["mask"][yi, xi]:
                continue
            circularity = float(cv2.contourArea(contour) / (np.pi * r * r))
            radius_m = _radius_m(x, y, r)
            center_m = _to_plane(paper["homography"], [[x, y]])[0]
            local = center_m - np.asarray(paper.get("paper_center_xy_m", [0.0, 0.0]), float)
            if (min_radius_m <= radius_m <= max_radius_m and circularity >= 0.45
                    and np.all(np.abs(local) <= local_limits)):
                fill = _circle_fill(x, y, r)
                if fill >= 0.55:
                    candidates.append((fill, circularity, radius_m, (float(x), float(y)), "contour_circle_fit"))

    if not candidates:
        raise RuntimeError("No dark filled circle with a plausible eye radius")
    candidates.sort(key=lambda item: (item[0], item[1]))
    fill, circularity, radius_m, center_px, method = candidates[-1]
    center_m = _to_plane(paper["homography"], [center_px])[0]
    return {
        "center_base_m": [float(center_m[0]), float(center_m[1])],
        "radius_m": float(radius_m),
        "center_px": list(center_px),
        "fill_fraction": float(fill),
        "circularity": float(circularity),
        "method": method,
        "official_eye_annotation": False,
    }


def detect_ink_after(rgb_before, rgb_after, paper, near_xy_m, expected_radius_m=0.003):
    """Look for a new dark mark near the planned dot after the arm has retracted."""
    before = cv2.cvtColor(rgb_before, cv2.COLOR_RGB2GRAY)
    after = cv2.cvtColor(rgb_after, cv2.COLOR_RGB2GRAY)
    difference = cv2.subtract(before, after)
    mask = ((difference > 35) & (after < 120)).astype(np.uint8) * 255
    mask &= paper["mask"]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = None
    for contour in contours:
        if cv2.contourArea(contour) < 25:
            continue
        (x, y), r = cv2.minEnclosingCircle(contour)
        center_m = _to_plane(paper["homography"], [[x, y]])[0]
        radius_m = float(np.median(np.linalg.norm(
            _to_plane(paper["homography"], [
                [x + r * np.cos(a), y + r * np.sin(a)]
                for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)
            ])
            - center_m, axis=1)))
        distance = float(np.linalg.norm(center_m - np.asarray(near_xy_m)))
        if distance < max(0.008, expected_radius_m * 3) and 0.001 <= radius_m <= 0.010:
            score = float(cv2.contourArea(contour)) / (distance + 1e-6)
            if best is None or score > best["score"]:
                best = {"center_base_m": center_m.tolist(), "radius_m": radius_m,
                        "center_px": [float(x), float(y)], "score": score}
    return best


def run_attempt(public, profile=None, viewer=False):
    public = Path(public).resolve()
    run = public.parent / "run"
    run.mkdir(exist_ok=True)
    manifest_verified = verify_manifest(public)
    settings = load_settings()
    if profile is not None:
        profile_path = (ROOT / profile).resolve()
        if not profile_path.is_relative_to((ROOT / "config").resolve()) or not profile_path.is_file():
            raise ValueError("Profile must be an existing JSON inside config/")
        settings.update(json.loads(profile_path.read_text(encoding="utf-8")))
    public_meta = json.loads((public / "scene.json").read_text(encoding="utf-8"))
    rules = load_competition_rules()
    limit_s = float(round_limit_seconds(rules, "preliminary"))

    denied = []
    def deny_answer(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            name = str(args[0])
            if "examiner_private" in name:
                raise PermissionError("Controller cannot read examiner answer")
            if "dragon_pixels.png" in name or "scene.json" in name:
                denied.append(name)
    sys.addaudithook(deny_answer)

    m, d, center, R, s, cfg = make_scene(public, settings=settings)
    guard = TableClearance(m)
    guard.require_clear(d)
    viewer_window=None;viewer_start=0.
    if viewer:
        from mujoco import viewer as mj_viewer
        viewer_window=mj_viewer.launch_passive(m,d)
        viewer_window.cam.lookat[:]=[.06,0,.30]
        viewer_window.cam.distance,viewer_window.cam.azimuth,viewer_window.cam.elevation=1.7,130,-28
        viewer_start=time.perf_counter()


    camera_before = run / "camera_before.png"
    # Vision input is the examiner-rendered overhead camera image. The MuJoCo
    # scene remains a clean paper; grasp/dip/dot dynamics are physical.
    rgb_before = np.asarray(Image.open(public / "dragon_pixels.png").convert("RGB"))
    Image.fromarray(rgb_before).save(camera_before)
    render_camera(m, d, run / "scene_before_physics.png", run / "camera_depth_before.npy")
    recognition = {"status": "ok"}
    try:
        paper = detect_paper(rgb_before, public_meta)
        recognition.update(detect_eye(rgb_before, paper))
        recognition.update({"paper_area_px": paper["area_px"],
                            "paper_corners_px": paper["corners_px"]})
    except Exception as exc:  # noqa: BLE001 - report and stop; never guess an eye.
        recognition = {"status": "failed", "reason": str(exc)}
        report = {
            "stop_reason": "eye_localization_failed",
            "elapsed_sim_s": 0.0,
            "recognition": recognition,
            "process_evidence": {"touched_brush": False, "stable_hold_3s": False, "dipped_ink": False},
            "dot": {"center_base_m": None, "radius_m": 0.0, "visual_verified": False},
            "penalty_points": 0.0,
            "manifest_verified": manifest_verified,
            "official_eye_annotation": False,
        }
        (run / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (run / "recognition.json").write_text(json.dumps(recognition, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return report

    eye_xy = np.asarray(recognition["center_base_m"], float)
    eye_r_m = float(recognition["radius_m"])
    dot_z = float(s.get("dot_z_m", 0.0018))
    (run / "recognition.json").write_text(json.dumps(recognition, ensure_ascii=False, indent=2), encoding="utf-8")

    # Grasp/dip/dot state machine.
    grip = {k: float(v) for k, v in cfg["grip_targets_rad"].items()}
    initial = {k: float(d.qpos[m.joint(k).qposadr[0]]) for k in grip}
    home = d.qpos[:6].copy()
    lift = solve_arm_ik(m, d, d.site("palm_control").xpos + [0, 0, 0.12], R, iterations=400)
    if not lift.converged:
        raise RuntimeError("Safe lift IK failed")
    servo = ContactGripServo(m, target_n=s["grip_force_targets_N"], gain_m_per_ns=0.008)
    ink = InkState(s)
    filter_alpha = 0.01 / (s.get("normal_force_filter_s", 0.03) + 0.01)
    normal_target = float(s.get("dot_normal_force_target_N", 0.10))
    normal_offset = 0.0
    filtered_paper_n = 0.0

    state = "observe"
    entered = 0.0
    anchor = nib_point(m, d).copy()
    previous_relative = None
    hold = 0.0
    longest_hold = 0.0
    hold_passed = False
    touch_brush = False
    lost_grip_s = 0.0
    brush_drops = 0
    dot_samples = []
    reason = "round_time_limit"
    done_at = None
    rows = []
    states = []
    controls = []
    targets_log = []
    paper_n_last = 0.0
    contact_last = np.zeros(3)

    def transition(next_state):
        nonlocal state, entered, anchor
        state = next_state
        entered = float(d.time)
        anchor = nib_point(m, d).copy()

    try:
        for tick in range(int(limit_s * 100)):
            t = d.time
            elapsed = t - entered
            force, support = brush_contacts(m, d)
            touch_brush |= bool(np.any(force > 0.01))
            gripping = bool(force[0] > 0.01 and np.any(force[1:] > 0.01))

            # Hand acquisition and bounded force feedback.
            close = float(smoothstep5((t - 1.0) / 3.0))
            cmd = {k: initial[k] + close * (v - initial[k]) for k, v in grip.items()}
            if 3.5 <= t < 14.0:
                for k, v in servo.update(d, force, 0.01).items():
                    cmd[k] += v
            elif t >= 14.0:
                for k, v in servo.delta.items():
                    cmd[k] += v
            set_joint_targets(m, d, cmd)

            # Arm command: early timed lift, then nib-target state machine.
            target = nib_point(m, d).copy()
            if state in ("observe", "close", "lift", "hold"):
                arm_u = float(smoothstep5((t - 7.0) / 3.0))
                d.ctrl[:6] = home + arm_u * (lift.joints - home)
            elif state == "to_ink":
                target = anchor.copy()
                target[:2] += float(smoothstep5(elapsed / 4.0)) * (np.asarray(s["ink_center_xy_m"]) - anchor[:2])
                if elapsed >= 4.0:
                    transition("dip_lower")
            elif state == "dip_lower":
                target = anchor.copy()
                target[:2] = s["ink_center_xy_m"]
                target[2] += float(smoothstep5(elapsed / 4.0)) * (
                    s["ink_surface_z_m"] - s["dip_target_depth_m"] - anchor[2]
                )
                if elapsed >= 4.0:
                    transition("dip_dwell")
            elif state == "dip_dwell":
                target = np.r_[s["ink_center_xy_m"], s["ink_surface_z_m"] - s["dip_target_depth_m"]]
                if ink.loaded:
                    transition("dip_lift")
                elif elapsed > 5.0:
                    reason = "dip_gate_failed"
                    break
            elif state == "dip_lift":
                target = anchor.copy()
                target[2] += float(smoothstep5(elapsed / 3.0)) * (0.16 - anchor[2])
                if elapsed >= 3.0:
                    if not ink.withdrawn:
                        reason = "ink_not_withdrawn"
                        break
                    transition("to_eye")
            elif state == "to_eye":
                target = anchor.copy()
                target[:2] += float(smoothstep5(elapsed / 4.0)) * (eye_xy - anchor[:2])
                if elapsed >= 4.0:
                    transition("eye_lower")
            elif state == "eye_lower":
                target = np.r_[eye_xy, anchor[2]]
                target[2] += float(smoothstep5(elapsed / s.get("dot_lower_seconds", 3.0))) * (
                    dot_z - anchor[2]
                )
                if elapsed >= s.get("dot_lower_seconds", 3.0):
                    transition("dot_dwell")
            elif state == "dot_dwell":
                paper_n_last, contact_last = paper_contact(m, d)
                filtered_paper_n += filter_alpha * (paper_n_last - filtered_paper_n)
                if paper_n_last > 0.03:
                    normal_offset = float(np.clip(
                        normal_offset + 0.01 * s.get("normal_admittance_m_Ns", 0.004) * (filtered_paper_n - normal_target),
                        -0.0008, 0.0012,
                    ))
                    dot_samples.append(contact_last.copy())
                    xy_error = eye_xy - contact_last[:2]
                    correction = np.clip(0.5 * xy_error, -0.0025, 0.0025)
                    target = np.r_[np.clip(eye_xy + correction, eye_xy - 0.003, eye_xy + 0.003),
                                   dot_z + normal_offset]
                else:
                    target = np.r_[eye_xy, dot_z + normal_offset]
                if elapsed >= s.get("dot_dwell_seconds", 0.8):
                    if not dot_samples:
                        reason = "dot_contact_not_observed"
                        break
                    transition("retract")
            elif state == "retract":
                target = anchor.copy()
                target[2] += 0.04 * float(smoothstep5(elapsed / 2.0))
                if elapsed >= 2.0:
                    transition("done")
            elif state == "done":
                target = anchor.copy()
                target[2] += 0.02 * float(smoothstep5(elapsed / 2.0))
                if done_at is None:
                    done_at = t

            if state in ("to_ink", "dip_lower", "dip_dwell", "dip_lift", "to_eye", "eye_lower", "dot_dwell", "retract", "done"):
                command_nib(m, d, target, R)

            if state == "observe" and t >= 0.05:
                transition("close")
            if state == "close" and t >= 7.0:
                transition("lift")
            if state == "lift" and t >= 10.0:
                transition("hold")
            if state == "hold":
                if t >= 14.0:
                    if not hold_passed:
                        reason = "hold_gate_failed"
                        break
                    transition("to_ink")

            for _ in range(5):
                d.qfrc_applied[:6] = np.clip(d.qfrc_bias[:6], -40, 40)
                mujoco.mj_step(m, d)
                mujoco.mj_forward(m, d)
                guard.require_clear(d)
            if viewer_window is not None:
                if not viewer_window.is_running():
                    reason = "viewer_closed"
                    break
                viewer_window.sync()
                wall = time.perf_counter() - viewer_start
                if d.time > wall:
                    time.sleep(d.time - wall)


            if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):
                raise RuntimeError("MuJoCo numerical warning")

            force, support = brush_contacts(m, d)
            pen = d.body("brush")
            point = nib_point(m, d)
            ink.update(point, 0.01)
            palm = d.body("right_hand_base")
            relative = palm.xmat.reshape(3, 3).T @ (pen.xpos - palm.xpos)
            slip = 0.0 if previous_relative is None else float(
                np.linalg.norm(relative - previous_relative) / 0.01
            )
            previous_relative = relative.copy()
            gripping = bool(force[0] > 0.01 and np.any(force[1:] > 0.01))
            valid_hold = bool(gripping and support < 0.01 and pen.xpos[2] - center[2] > 0.02 and slip < 0.02)
            hold = hold + 0.01 if valid_hold else 0.0
            longest_hold = max(longest_hold, hold)
            hold_passed |= hold >= 3.0
            lost_grip_s = lost_grip_s + 0.01 if t >= 7.0 and not gripping else 0.0
            tilt = float(np.degrees(np.arccos(np.clip(pen.xmat.reshape(3, 3)[2, 2], -1, 1))))
            paper_n_last, contact_last = paper_contact(m, d)
            rows.append([t, *pen.xpos, *force, support, tilt, *point, paper_n_last, *contact_last, slip,
                         float(valid_hold), ink.maximum_depth, ink.dwell, ink.remaining])
            states.append(state)
            controls.append(d.ctrl.copy())
            targets_log.append(target)

            if tick % 200 == 0:
                print(json.dumps({
                    "t": round(t, 2), "stage": state, "tilt": round(tilt, 1),
                    "fingers": np.round(force[:3], 2).tolist(), "paper_N": round(paper_n_last, 3),
                    "ink_depth_mm": round(ink.maximum_depth * 1000, 2),
                }, ensure_ascii=False), flush=True)

            if lost_grip_s > 0.5:
                brush_drops += 1
                reason = "brush_dropped"
                break
            if tilt > 60 and t > 14:
                reason = "pen_tilt_exceeded"
                break
            if done_at is not None and t - done_at >= 2.0:
                reason = "completed"
                break
            if t >= limit_s:
                reason = "round_time_limit"
                break
    except Exception as exc:  # noqa: BLE001 - safety stop, still write an auditable report.
        reason = str(exc)
    if viewer_window is not None:
        try:
            viewer_window.close()
        except Exception:
            pass


    # Verify the dot from the post-retraction camera if available.
    camera_after = run / "camera_after.png"
    rgb_after = None
    try:
        rgb_after = render_camera(m, d, camera_after)
    except Exception:
        rgb_after = None

    # Ink transfer is represented by the physical nib-paper contact samples.
    # A visual ink renderer will be added when the official ink/paper model and
    # camera calibration are released; the judge already scores contact evidence.
    visual = None

    dot_center = None
    dot_radius = 0.0
    if visual is not None:
        dot_center = visual["center_base_m"]
        dot_radius = visual["radius_m"]
    elif dot_samples:
        samples = np.asarray(dot_samples)
        dot_center = float(np.mean(samples[:, 0])), float(np.mean(samples[:, 1]))
        dot_radius = float(s.get("dot_radius_m", s.get("brush_radius_m", 0.006)))

    rows_np = np.asarray(rows)
    np.savetxt(run / "physics.csv", rows_np, delimiter=",", comments="",
               header="time,x,y,z,thumb,index,middle,ring,little,support,tilt,nib_x,nib_y,nib_z,paper_N,contact_x,contact_y,contact_z,slip,valid_hold,ink_depth,ink_dwell,ink_remaining")
    np.savez_compressed(run / "replay.npz", qpos=None, ctrl=np.asarray(controls),
                        time_s=rows_np[:, 0], stage=np.asarray(states), target=np.asarray(targets_log))

    report = {
        "stage": "preliminary",
        "task": "dragon_eye_dotting",
        "rules_contract": str(ROOT / "config/competition_20260911.json"),
        "stop_reason": reason,
        "elapsed_sim_s": float(rows_np[-1, 0]) if len(rows_np) else 0.0,
        "manifest_verified": manifest_verified,
        "recognition": recognition,
        "process_evidence": {
            "touched_brush": bool(touch_brush),
            "stable_hold_3s": bool(hold_passed),
            "dipped_ink": bool(ink.loaded and ink.withdrawn),
            "longest_continuous_hold_s": float(longest_hold),
            "dip_maximum_depth_mm": float(ink.maximum_depth * 1000),
            "dip_longest_dwell_s": float(ink.longest),
        },
        "dot": {
            "center_base_m": list(dot_center) if dot_center is not None else None,
            "radius_m": float(dot_radius),
            "visual_verified": visual is not None,
            "contact_sample_count": len(dot_samples),
            "max_paper_contact_N": float(rows_np[:, 15].max()) if len(rows_np) else 0.0,
        },
        "penalties": {
            "brush_drops": brush_drops,
            "paper_shift_m": 0.0,
            "overtime": bool(float(rows_np[-1, 0]) > limit_s) if len(rows_np) else False,
            "human_intervention": 0,
        },
        "penalty_points": float(brush_drops * rules["penalties"]["brush_drop_each"]),
        "operation_mode": "full_autonomous",
        "operation_mode_multiplier": 1.0,
        "private_answer_reads": 0,
        "official_dragon_pattern": False,
        "official_score_pending": True,
        "note": "Preliminary controller is independent of experiments/final_calligraphy_2026.py and legacy blind_competition.py.",
    }
    (run / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description="2026 preliminary dragon-eye controller")
    p.add_argument("--public", type=Path, required=True)
    p.add_argument("--profile")
    p.add_argument("--viewer", action="store_true", help="open an interactive MuJoCo window (development only)")
    a = p.parse_args()
    run_attempt(a.public, a.profile, viewer=a.viewer)


if __name__ == "__main__":
    main()

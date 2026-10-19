import argparse
import json
import platform
import time
from importlib import metadata
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .control import set_joint_targets, solve_arm_ik, tactile_forces
from .paths import ASSETS, output_dir
from .scene import load_scene
from .trajectory import make_path, smoothstep5
from .vision import OfflineOCR


def json_out(filename, value):
    content = json.dumps(value, ensure_ascii=False, indent=2)
    (output_dir() / filename).write_text(content, encoding="utf-8")
    print(content)


def load_workcell():
    model = load_scene()
    data = mujoco.MjData(model)
    for i, angle in enumerate([0, 1.6, 1.6, 1.6, 1.57, 0], 1):
        data.qpos[model.joint(f"r_a_joint{i}").qposadr[0]] = angle
    for finger in range(1, 6):
        for joint in (2, 3, 4):
            data.qpos[model.joint(f"r_f_joint{finger}_{joint}").qposadr[0]] = .12
    mujoco.mj_forward(model, data)
    targets = {model.joint(i).name: data.qpos[model.jnt_qposadr[i]]
               for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE}
    set_joint_targets(model, data, targets)
    return model, data


def camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [.06, 0, .30]
    cam.distance, cam.azimuth, cam.elevation = 1.7, 130, -28
    return cam


def doctor():
    model, _ = load_workcell()
    json_out("environment_report.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: metadata.version(p) for p in ["mujoco", "numpy", "scipy", "opencv-python", "rapidocr-onnxruntime", "onnxruntime"]},
        "model_loaded": True, "nq": model.nq, "nv": model.nv, "actuators": model.nu,
        "sensor_values": model.nsensordata, "arm_joint_coordinates": 6,
        "hand_joint_coordinates": 20, "free_brush": True,
        "physical_hardware_validated": False, "exact_competition_model_confirmed": False,
    })


def demo(seconds=6., headless=True):
    model, data = load_workcell()
    trace = []
    viewer = None
    start = time.perf_counter()
    if not headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = camera().lookat
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.7, 130, -28
    try:
        for step in range(int(seconds / model.opt.timestep)):
            if viewer is not None and not viewer.is_running():
                break
            phase = (data.time % 6) / 3
            curl = .12 + .3 * float(smoothstep5(phase if phase <= 1 else 2-phase))
            set_joint_targets(model, data, {f"r_f_joint{f}_{j}": curl for f in range(1, 6) for j in (2, 3, 4)})
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Non-finite simulation state")
            if step % 10 == 0:
                trace.append([data.time, *tactile_forces(model, data), float(np.max(np.abs(data.qvel)))])
                if viewer is not None:
                    viewer.sync()
                    time.sleep(max(0., data.time-(time.perf_counter()-start)))
        wall = time.perf_counter()-start
        trace = np.asarray(trace)
        np.savetxt(output_dir()/"simulation_trace.csv", trace, delimiter=",", comments="",
                   header="time_s,thumb_n,index_n,middle_n,ring_n,little_n,max_abs_qvel")
        with mujoco.Renderer(model, height=960, width=1280) as renderer:
            renderer.update_scene(data, camera=camera())
            Image.fromarray(renderer.render()).save(output_dir()/"workcell.png")
            renderer.update_scene(data, camera="overhead")
            Image.fromarray(renderer.render()).save(output_dir()/"overhead.png")
        warnings = {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number}
        json_out("simulation_report.json", {
            "simulated_seconds": data.time, "wall_seconds_excluding_render": wall,
            "real_time_factor_excluding_render": data.time/max(wall, 1e-9),
            "finite_state": True, "warnings": warnings,
            "max_sampled_fingertip_force_n": float(trace[:, 1:6].max()) if len(trace) else None,
            "rendered": True, "demo_behavior": "position-controlled finger opening and closing",
            "autonomous_brush_pickup_validated": False, "physical_hardware_validated": False,
        })
        if warnings:
            raise RuntimeError(f"MuJoCo warnings: {warnings}")
    finally:
        if viewer is not None:
            viewer.close()


def plan(text):
    path = make_path(text)
    np.savetxt(output_dir()/"pen_path.csv", np.c_[path.time_s, path.xyz, path.pen_down.astype(int)],
               delimiter=",", header="time_s,x_m,y_m,z_m,pen_down", comments="")
    xy = path.xyz[:, :2]
    span = np.ptp(xy, axis=0)
    scale = min(1080/max(span[0], .001), 650/max(span[1], .001))
    uv = (xy-xy.min(0))*scale
    uv[:, 1] = uv[:, 1].max()-uv[:, 1]
    uv += [60, 80]
    canvas = Image.new("RGB", (1200, 800), "#faf8ef")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 25), "PEN-TIP PATH PREVIEW | Not robot-written ink | SI units in CSV", fill="#526052")
    for i in range(1, len(uv)):
        if path.pen_down[i]:
            draw.line([tuple(uv[i-1]), tuple(uv[i])], fill="#19231d", width=4)
    canvas.save(output_dir()/"pen_path_preview.png")
    json_out("trajectory_report.json", {
        "text": text, "samples": len(path.xyz), "planned_duration_s": float(path.time_s[-1]),
        "pen_up_samples": int((~path.pen_down).sum()), "units": "meters, seconds",
        "source": "Hanzi Writer Data / Make Me a Hanzi stroke medians",
        "pen_down_speed_limit_m_s": .05, "pen_up_speed_limit_m_s": .15,
        "final_round_budget_s": 300., "reserved_for_grip_ink_and_perception_s": 60.,
        "fits_trajectory_budget": bool(path.time_s[-1] <= 240.),
        "force_profile_calibrated": False, "executed_by_robot": False,
    })


def ocr(image_path=None):
    if image_path is None:
        sample = Image.new("RGB", (900, 260), "white")
        ImageDraw.Draw(sample).text((70, 70), "静夜思", font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 80), fill="black")
        image_path = output_dir()/"ocr_input.png"
        sample.save(image_path)
    bgr = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot decode image: {image_path}")
    engine = OfflineOCR()
    start = time.perf_counter()
    results = engine.read(bgr)
    json_out("ocr_report.json", {"image": str(image_path), "results": results,
             "inference_seconds": time.perf_counter()-start, "backend": "ONNX Runtime CPU", "offline": True})
    return results


def ik_check():
    model, data = load_workcell()
    site = model.site("palm_control").id
    target = data.site_xpos[site] + [.015, -.01, .015]
    result = solve_arm_ik(model, data, target, data.site_xmat[site].reshape(3, 3).copy())
    json_out("ik_report.json", {"converged": result.converged, "position_error_m": result.position_error_m,
             "orientation_error_rad": result.orientation_error_rad, "joints_rad": result.joints.tolist(),
             "target_position_m": target.tolist(), "collision_checked": False, "executed": False})
    if not result.converged:
        raise RuntimeError("Reference IK target did not converge")


def prepare(image_path=None):
    """Connect offline title recognition to a fixed, reviewable local poem database."""
    results = ocr(image_path)
    titles = {item["text"].strip(" 《》\n\r\t") for item in results if item["confidence"] >= .85}
    poems = json.loads((ASSETS/"poems.json").read_text(encoding="utf-8"))["poems"]
    matches = [p for p in poems if p["title"] in titles]
    if len(matches) != 1:
        raise ValueError("Need one confident title matching the local poem database; no poem was invented")
    poem = matches[0]
    plan("".join(poem["lines"]))
    json_out("poem_report.json", {"poem": poem, "matched_offline_database": True,
             "official_question_bank": False, "robot_execution_started": False})
    (output_dir()/"poem_to_read.txt").write_text(poem["title"]+"。"+"。".join(poem["lines"])+"。", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Five-finger simulation and algorithm baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sim = sub.add_parser("demo")
    sim.add_argument("--seconds", type=float, default=6.)
    sim.add_argument("--headless", action="store_true")
    view = sub.add_parser("viewer")
    view.add_argument("--seconds", type=float, default=3600.)
    stroke = sub.add_parser("plan")
    stroke.add_argument("--text", default="永")
    recognition = sub.add_parser("ocr")
    recognition.add_argument("--image", type=Path)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--image", type=Path)
    sub.add_parser("ik-check")
    args = parser.parse_args()
    if hasattr(args, "seconds") and (not np.isfinite(args.seconds) or args.seconds < .02):
        parser.error("--seconds must be finite and at least 0.02")
    if args.command == "doctor": doctor()
    elif args.command in ("demo", "viewer"): demo(args.seconds, headless=getattr(args, "headless", False))
    elif args.command == "plan": plan(args.text)
    elif args.command == "ocr": ocr(args.image)
    elif args.command == "ik-check": ik_check()
    elif args.command == "prepare": prepare(args.image)


if __name__ == "__main__":
    main()

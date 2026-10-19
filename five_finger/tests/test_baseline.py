import mujoco
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from fivefinger.cli import load_workcell
from fivefinger.control import ForceLimitedClosure, set_joint_targets, solve_arm_ik
from fivefinger.task import Observation, Stage, TaskMachine
from fivefinger.trajectory import make_path
from fivefinger.vision import OfflineOCR, PlaneCalibration


def test_measured_plane_mapping_and_reject_degenerate_calibration():
    calibration = PlaneCalibration.from_correspondences(
        [[100, 100], [800, 100], [800, 600], [100, 600]],
        [[0, 0], [.42, 0], [.42, .297], [0, .297]])
    assert np.allclose(calibration.to_plane([[450, 350]]), [[.21, .1485]], atol=1e-7)
    with pytest.raises(ValueError):
        PlaneCalibration.from_correspondences([[0, 0], [1, 1], [2, 2], [3, 3]], [[0, 0], [1, 1], [2, 2], [3, 3]])


def test_metric_strokes_speed_and_pen_clearance():
    path = make_path("永")
    assert np.isfinite(path.xyz).all()
    assert len(path.xyz) == len(path.pen_down) == len(path.time_s)
    assert np.allclose(path.xyz[[0, -1], 2], .015)
    assert path.pen_down.any() and (~path.pen_down).any()
    velocity = np.linalg.norm(np.diff(path.xyz, axis=0), axis=1) / np.diff(path.time_s)
    assert np.max(velocity[path.pen_down[1:]]) <= .050001
    assert np.max(velocity[~path.pen_down[1:]]) <= .150001
    assert np.allclose(path.xyz[path.pen_down, 2], 0.)
    with pytest.raises(ValueError, match="No offline stroke data"):
        make_path("🙂")


def test_sample_poem_fits_planned_time_budget():
    path = make_path("床前明月光疑是地上霜举头望明月低头思故乡")
    assert path.time_s[-1] <= 240.


def test_task_requires_continuous_grip_and_feedback():
    timed_out = TaskMachine(limit_s=2.)
    assert timed_out.update(Observation(), 2.) == Stage.FAILED
    task = TaskMachine()
    task.update(Observation(target_valid=True), .01)
    task.update(Observation(approach_reached=True), .01)
    assert task.update(Observation(brush_contact=True, brush_held=True), .01) == Stage.HOLD
    assert task.update(Observation(brush_held=True), 2.9) == Stage.HOLD
    assert task.update(Observation(brush_held=False), .1) == Stage.GRIP
    task.update(Observation(brush_contact=True, brush_held=True), .01)
    assert task.update(Observation(brush_held=True), 3.) == Stage.DIP
    assert task.update(Observation(brush_held=True, ink_depth_m=.004), .6) == Stage.DIP
    assert task.update(Observation(brush_held=True, ink_depth_m=.004, ink_visible=True), .01) == Stage.TRANSFER
    task.update(Observation(brush_held=True, paper_reached=True), .01)
    assert task.update(Observation(brush_held=True, action_finished=True), .01) == Stage.EXECUTE
    assert task.update(Observation(brush_held=True, action_finished=True, result_verified=True), .01) == Stage.DONE


def test_tactile_controller_releases_excessive_contact():
    control = ForceLimitedClosure()
    control.update(np.zeros(5), 1.)
    before = control.curl.copy()
    control.update(np.ones(5)*3., .1)
    assert (control.curl < before).all()


def test_vendor_model_pose_ik_and_finite_step():
    model, data = load_workcell()
    site = model.site("palm_control").id
    original = data.qpos.copy()
    result = solve_arm_ik(model, data, data.site_xpos[site]+[.015, -.01, .015], data.site_xmat[site].reshape(3, 3).copy())
    assert result.converged, result
    assert np.array_equal(data.qpos, original)
    set_joint_targets(model, data, {"r_f_joint2_2": 100.})
    aid = model.actuator("act_r_f_joint2_2").id
    assert data.ctrl[aid] == model.actuator_ctrlrange[aid, 1]
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert not any(w.number for w in data.warning)


def test_offline_ocr_recognizes_chinese_title():
    canvas = Image.new("RGB", (900, 260), "white")
    ImageDraw.Draw(canvas).text((70, 70), "静夜思", font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 80), fill="black")
    results = OfflineOCR().read(np.asarray(canvas)[:, :, ::-1].copy())
    assert "静夜思" in "".join(item["text"] for item in results)

"""Record actual MuJoCo steps to H.264, with wide and close-up camera views."""
import json
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fivefinger.cli import load_workcell
from fivefinger.control import set_joint_targets
from fivefinger.paths import output_dir
from fivefinger.trajectory import smoothstep5


def main():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found on PATH")
    output = output_dir()
    video = output / "five_finger_simulation.mp4"
    width, height, fps, seconds = 1280, 720, 30, 12
    model, data = load_workcell()
    font_path = "C:/Windows/Fonts/msyh.ttc"
    title_font = ImageFont.truetype(font_path, 25)
    text_font = ImageFont.truetype(font_path, 19)
    camera = mujoco.MjvCamera()
    selected = []
    with (output / "video_encode.log").open("wb") as errors:
        encoder = subprocess.Popen([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            with mujoco.Renderer(model, height=height, width=width) as renderer:
                for frame in range(seconds * fps):
                    target_time = frame/fps
                    while data.time + model.opt.timestep/2 < target_time:
                        phase = (data.time % 6) / 3
                        curl = .12 + .3 * float(smoothstep5(phase if phase <= 1 else 2-phase))
                        set_joint_targets(model, data, {f"r_f_joint{f}_{j}": curl for f in range(1, 6) for j in (2, 3, 4)})
                        mujoco.mj_step(model, data)
                    if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
                        raise RuntimeError("Simulation state failed validation during recording")
                    close_up = target_time >= 6.
                    camera.lookat[:] = [.13, .09, .21] if close_up else [.08, 0., .22]
                    camera.distance = .52 if close_up else 1.4
                    camera.azimuth, camera.elevation = 130., -18. if close_up else -26.
                    renderer.update_scene(data, camera=camera)
                    picture = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(picture)
                    draw.rectangle((0, 0, width, 57), fill="#18252d")
                    draw.text((23, 12), "MuJoCo 五指物理仿真  |  " + ("五指近景" if close_up else "场景全景"), font=title_font, fill="white")
                    draw.text((1040, 18), f"{target_time:05.2f} s / 12 s", font=text_font, fill="#b9d7e7")
                    draw.rectangle((0, height-46, width, height), fill="#18252d")
                    draw.text((23, height-36), "当前演示：五指张合；自动抓笔、蘸墨和书写尚未接入。", font=text_font, fill="#e0e8ed")
                    encoder.stdin.write(picture.tobytes())
                    if frame in (0, 90, 180, 270, 359):
                        selected.append(picture.copy())
            encoder.stdin.close()
            if encoder.wait(timeout=30) != 0:
                raise RuntimeError("Video encoding failed; see outputs/video_encode.log")
        except BaseException:
            encoder.kill()
            encoder.wait()
            raise
    sheet = Image.new("RGB", (1280, 720*len(selected)), "white")
    for i, picture in enumerate(selected):
        sheet.paste(picture, (0, i*720))
    sheet.save(output / "video_contact_sheet.jpg", quality=90)
    selected[0].save(output / "video_poster.jpg", quality=95)
    report = {"file": str(video), "fps": fps, "frames": fps*seconds, "duration_s": seconds,
              "resolution": [width, height], "source": "MuJoCo physical simulation steps",
              "behavior": "finger opening and closing; fixed arm position", "autonomous_writing": False,
              "finite_simulation_state": True, "mujoco_warnings": []}
    (output / "video_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

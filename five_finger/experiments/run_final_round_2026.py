"""Run one 2026 final calligraphy round.

Draw a random poem title -> freeze the controller -> run the four-A3 2x2 scene ->
judge with the revised scoring model. Video export is intentionally left to the
legacy viewer until the new case layout has its own renderer.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from fivefinger.paths import ROOT

CASES = ROOT / "outputs" / "final_2026"


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", check=False)


def main():
    p = argparse.ArgumentParser(description="2026 final calligraphy round runner")
    p.add_argument("--development-title", help="Use a known local poem title instead of a random draw")
    p.add_argument("--profile", help="Optional pre-draw profile JSON under config/")
    p.add_argument("--viewer", action="store_true", help="open an interactive MuJoCo window (development only)")
    a = p.parse_args()

    draw_cmd = [sys.executable, "-m", "experiments.final_challenge_2026", "create"]
    if a.development_title:
        draw_cmd += ["--development-title", a.development_title]
    draw = _run(draw_cmd)
    if draw.returncode:
        raise RuntimeError(f"Final draw failed:\n{draw.stdout}\n{draw.stderr}")
    case_id = json.loads(draw.stdout)["case_id"]
    folder = CASES / case_id
    print(draw.stdout, flush=True)

    controller_cmd = [sys.executable, "-m", "experiments.final_calligraphy_2026",
                      "--public", str(folder / "public")]
    if a.profile:
        controller_cmd += ["--profile", a.profile]
    if a.viewer:
        controller_cmd += ["--viewer"]
    controller = _run(controller_cmd)
    (folder / "controller.log").write_text(controller.stdout, encoding="utf-8")
    print(controller.stdout, flush=True)
    if controller.returncode:
        (folder / "run/failure.json").write_text(json.dumps({
            "returncode": controller.returncode,
            "stderr": controller.stderr[-6000:],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    judge = _run([sys.executable, "-m", "experiments.final_challenge_2026", "judge", "--case", case_id])
    if judge.returncode:
        raise RuntimeError(f"Final judge failed:\n{judge.stdout}\n{judge.stderr}")
    print(judge.stdout, flush=True)

    (CASES / "latest_case.txt").write_text(case_id, encoding="utf-8")
    if controller.returncode:
        raise SystemExit(controller.returncode)


if __name__ == "__main__":
    main()

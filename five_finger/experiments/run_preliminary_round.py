"""Run one full 2026 preliminary round: up to three independent 180 s attempts.

Each attempt gets a fresh practice dragon draw. The official preliminary total is
the best single attempt (time is taken from that best attempt). This runner keeps
the three cases under ``outputs/preliminary_2026`` and writes a round summary.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from fivefinger.competition_rules import best_preliminary_attempt, load_competition_rules
from fivefinger.paths import ROOT

CASES = ROOT / "outputs" / "preliminary_2026"


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", check=False)


def run_round(attempts: int = 3, viewer: bool = False) -> dict:
    rules = load_competition_rules()
    if not 1 <= attempts <= int(rules["preliminary"]["attempts"]):
        raise ValueError("Development runs may use 1..3 attempts; official rounds always use 3")

    round_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    round_dir = CASES / ("round_" + round_id)
    round_dir.mkdir(parents=True, exist_ok=False)
    attempts_scored = []

    for index in range(1, attempts + 1):
        draw = _run([sys.executable, "-m", "experiments.preliminary_challenge", "create"])
        if draw.returncode:
            raise RuntimeError(f"Attempt {index} draw failed:\n{draw.stdout}\n{draw.stderr}")
        case_id = json.loads(draw.stdout)["case_id"]
        case_dir = CASES / case_id

        controller = _run([
            sys.executable, "-m", "experiments.preliminary_dragon_eye",
            "--public", str(case_dir / "public"),
        ] + (["--viewer"] if viewer else []))
        (case_dir / "controller.log").write_text(controller.stdout, encoding="utf-8")
        if controller.returncode:
            (case_dir / "run/failure.json").write_text(json.dumps({
                "returncode": controller.returncode,
                "stderr": controller.stderr[-4000:],
            }, ensure_ascii=False, indent=2), encoding="utf-8")

        judge = _run([
            sys.executable, "-m", "experiments.preliminary_challenge", "judge",
            "--case", case_id,
        ])
        if judge.returncode:
            raise RuntimeError(f"Attempt {index} judging failed:\n{judge.stdout}\n{judge.stderr}")

        verdict = json.loads(judge.stdout) if judge.stdout.strip() else {}
        verdict["attempt_id"] = case_id
        verdict["attempt_index"] = index
        verdict["case_dir"] = str(case_dir)
        attempts_scored.append(verdict)
        (round_dir / f"attempt_{index}.json").write_text(
            json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"attempt": index, "case": case_id,
                          "raw_points": verdict.get("raw_points"),
                          "weighted_points": verdict.get("weighted_points")},
                         ensure_ascii=False), flush=True)

    best = best_preliminary_attempt(attempts_scored)
    summary = {
        "round_id": round_id,
        "stage": "preliminary",
        "attempts_run": attempts,
        "attempts_required_by_rules": rules["preliminary"]["attempts"],
        "best_attempt": best,
        "attempts": attempts_scored,
        "official_notes": "Official scoring still requires the referee to confirm physical process nodes and the official dragon-pattern eye geometry.",
    }
    (round_dir / "round_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (CASES / "latest_round.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "preliminary", "round_id": round_id,
                      "best_attempt": best}, ensure_ascii=False), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description="2026 preliminary round runner")
    p.add_argument("--attempts", type=int, default=3,
                   help="1..3; development smoke runs use 1")
    p.add_argument("--viewer", action="store_true",
                   help="open a MuJoCo window for each attempt (development only)")
    a = p.parse_args()
    run_round(a.attempts, viewer=a.viewer)


if __name__ == "__main__":
    main()

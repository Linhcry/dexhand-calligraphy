"""Independent 2026 final examiner: project a poem title, then score the run.

Like the legacy ``blind_challenge``, the controller only receives ``public``
pixels and published geometry. The 2026 final judge adds the revised four-A3
2x2 paper contract and the 0/1/2 + consecutive-five quality model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import time
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from experiments.final_scoring_2026 import evaluate_case
from fivefinger.paths import ROOT

CASES = ROOT / "outputs" / "final_2026"
FINAL_SETTINGS = ROOT / "config" / "competition_scene_2026_final.json"


def _make_manifest():
    files = list((ROOT / "fivefinger").rglob("*.py"))
    files += [
        ROOT / "experiments/final_challenge_2026.py",
        ROOT / "experiments/final_calligraphy_2026.py",
        ROOT / "experiments/run_final_round_2026.py",
        ROOT / "experiments/final_scoring_2026.py",
        ROOT / "experiments/blind_competition.py",
        ROOT / "experiments/competition_scene.py",
        ROOT / "experiments/writing.py",
        ROOT / "experiments/probe_grasp.py",
        ROOT / "experiments/ocr_camera_worker.py",
        ROOT / "experiments/writing_config.json",
        ROOT / "config/competition_scene_2026_final.json",
        ROOT / "config/competition_20260911.json",
        ROOT / "config/competition_scene.json",
        ROOT / "config/example_poems.json",
        ROOT / "scripts/speak_recognized.ps1",
    ]
    files += [p for p in (ROOT / "config").glob("*.json")]
    files += [p for p in (ROOT / "assets/ocr").glob("*") if p.is_file()]
    unique = sorted({p.resolve() for p in files if p.is_file()})
    return {
        str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in unique
    }


def create(development_title: str | None = None):
    bank = json.loads((ROOT / "config/example_poems.json").read_text(encoding="utf-8"))
    if development_title is not None and development_title not in bank:
        raise ValueError("Development title must be in the local poem bank")
    title = development_title or secrets.choice(list(bank))
    nonce = secrets.token_hex(32)
    case = CASES / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    public = case / "public"
    private = case / "examiner_private"
    public.mkdir(parents=True)
    private.mkdir()

    image = Image.new("RGB", (1000, 400), (237 + secrets.randbelow(15),) * 3)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 160)
    draw.text((500, 200), title, font=font, anchor="mm", fill=(18, 20, 25))
    image.save(public / "screen_pixels.png")

    scene_settings = json.loads(FINAL_SETTINGS.read_text(encoding="utf-8"))
    metadata = {
        "case_id": case.name,
        "evaluation_mode": "development_known_to_evaluator" if development_title else "random_blind_draw",
        "answer_commitment_sha256": hashlib.sha256((nonce + title + bank[title]).encode()).hexdigest(),
        "screen_pixels_sha256": hashlib.sha256((public / "screen_pixels.png").read_bytes()).hexdigest(),
        "screen_center_m": [0.60 + (secrets.randbelow(21) - 10) * 0.001,
                            0.28 + (secrets.randbelow(21) - 10) * 0.001, 0.12],
        "screen_yaw_deg": secrets.randbelow(13) - 6,
        "screen_tilt_deg": 35,
        "paper_layout": scene_settings["paper_layout"],
        "paper_size_m": scene_settings["paper_size_m"],
        "paper_center_xy_m": scene_settings["paper_center_xy_m"],
        "paper_sheets": scene_settings["paper_sheets"],
        "final_paper_contract": "four A3 sheets, 2x2 no gap, one character per grid",
        "examiner_note": "Non-official example poem bank; target sampled independently. Controller receives only this metadata and image bytes.",
    }
    (public / "scene.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (public / "controller_manifest.json").write_text(json.dumps(_make_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    (private / "answer.json").write_text(json.dumps({"title": title, "poem": bank[title], "nonce": nonce}, ensure_ascii=False, indent=2), encoding="utf-8")
    CASES.mkdir(exist_ok=True)
    (CASES / "latest_case.txt").write_text(case.name, encoding="utf-8")
    print(json.dumps({"case_id": case.name, "answer_commitment_sha256": metadata["answer_commitment_sha256"]}, ensure_ascii=False), flush=True)
    return case


def judge(case):
    folder = CASES / case
    secret = json.loads((folder / "examiner_private/answer.json").read_text(encoding="utf-8"))
    public = json.loads((folder / "public/scene.json").read_text(encoding="utf-8"))
    committed = hashlib.sha256((secret["nonce"] + secret["title"] + secret["poem"]).encode()).hexdigest() == public["answer_commitment_sha256"]
    report_path = folder / "run/report.json"
    if not report_path.exists():
        verdict = {"commitment_valid": committed, "status": "controller_failed_before_report",
                   "target_title": secret["title"], "target_poem": secret["poem"], "official_score": None}
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        evaluation = evaluate_case(folder, field_max_raw_quality_points=None)
        verdict = {
            "commitment_valid": committed,
            "recognized_title_matches": report.get("recognized_title") == secret["title"],
            "planned_body_matches": report.get("recognized_poem") == secret["poem"],
            "target_title": secret["title"],
            "target_poem": secret["poem"],
            "physical_result": report.get("stop_reason"),
            "elapsed_sim_s": report.get("elapsed_sim_s"),
            "elapsed_within_300s": bool(0.0 <= float(report.get("elapsed_sim_s", 300.1)) <= 300.0),
            "paper_contract_used": public.get("paper_layout") == "four_a3_2x2_no_gap",
            "quality_proxy": evaluation,
            "official_human_judge_required": True,
        }
    (folder / "judge_2026.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False), flush=True)
    return verdict


def main():
    p = argparse.ArgumentParser(description="2026 final calligraphy examiner")
    p.add_argument("action", choices=["create", "judge"])
    p.add_argument("--case")
    p.add_argument("--development-title")
    a = p.parse_args()
    if a.action == "create":
        create(a.development_title)
    else:
        judge(a.case or (CASES / "latest_case.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

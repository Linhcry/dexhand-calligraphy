"""2026 final calligraphy controller wrapper.

This entry point is intentionally separate from the legacy single-A3
``blind_competition`` controller and from ``preliminary_dragon_eye``. It runs the
shared physical grasp/dip/writing loop against the revised four-A3 2x2 scene,
then writes a development quality proxy. Official scoring is performed by
``final_challenge_2026.judge``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.blind_competition import run_case
from experiments.final_scoring_2026 import evaluate_case, load_settings
from fivefinger.paths import ROOT

FINAL_SETTINGS_RELPATH = "config/competition_scene_2026_final.json"


def _verify_final_scene_manifest(public: Path) -> bool:
    manifest_path = public / "controller_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if FINAL_SETTINGS_RELPATH not in manifest:
        raise RuntimeError(f"{FINAL_SETTINGS_RELPATH} is not frozen in the pre-draw manifest")
    settings_path = ROOT / FINAL_SETTINGS_RELPATH
    if hashlib.sha256(settings_path.read_bytes()).hexdigest() != manifest[FINAL_SETTINGS_RELPATH]:
        raise RuntimeError("Final scene settings changed after the random draw")
    return True


def main():
    p = argparse.ArgumentParser(description="2026 final calligraphy controller")
    p.add_argument("--public", type=Path, required=True)
    p.add_argument("--profile")
    p.add_argument("--viewer", action="store_true", help="open an interactive MuJoCo window (development only)")
    a = p.parse_args()
    public = a.public.resolve()
    manifest_verified = _verify_final_scene_manifest(public)
    settings = load_settings()
    run_case(public, profile=a.profile, settings=settings, viewer=a.viewer)
    folder = public.parent
    report = json.loads((folder / "run/report.json").read_text(encoding="utf-8"))
    try:
        evaluation = evaluate_case(folder, field_max_raw_quality_points=None)
    except Exception as exc:  # noqa: BLE001 - physics replay may be incomplete after a safety stop.
        evaluation = {"error": str(exc)}
    result = {
        "stage": "final",
        "paper_contract": settings["paper_layout"],
        "manifest_verified": manifest_verified,
        "legacy_physical_report": report,
        "quality_proxy_2026": evaluation,
        "note": "Final 2026 controller shares low-level physics skills with the legacy baseline but uses a dedicated 4-sheet scene and wrapper; preliminary uses a separate controller.",
    }
    (folder / "run/final_2026.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "final", "stop_reason": report.get("stop_reason"),
                      "elapsed_sim_s": report.get("elapsed_sim_s"),
                      "paper_contract": settings["paper_layout"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

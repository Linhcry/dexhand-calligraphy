"""Automatic development proxy for the 2026 final calligraphy quality score.

The official final score is assigned by a human judge panel. This module turns the
physics replay into transparent 0/1/2 character grades so the simulation can be
tested end-to-end before the official scoring sheet is available. It never claims
to be the official score.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fivefinger.competition_rules import (
    final_consecutive_bonus,
    final_raw_quality_points,
    load_competition_rules,
    score_final_round,
)
from fivefinger.paths import ROOT
from fivefinger.trajectory import make_path

FINAL_SETTINGS = ROOT / "config" / "competition_scene_2026_final.json"


def load_settings() -> dict[str, Any]:
    return json.loads(FINAL_SETTINGS.read_text(encoding="utf-8"))


def load_run_settings(case_dir: Path) -> dict[str, Any]:
    """Use the exact frozen scene/profile used by the physical run when available."""
    config_path = Path(case_dir) / "run/config.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if "scene" in payload:
            return payload["scene"]
    return load_settings()


def build_full_path(poem: str, settings: dict[str, Any] | None = None):
    s = load_settings() if settings is None else settings
    return make_path(
        poem,
        origin_xy=s["origin_xy_m"],
        character_size=s["character_size_m"],
        gap=s["character_gap_m"],
        landing_speed=s.get("landing_speed_m_s"),
        landing_band=s.get("landing_band_m", 0.002),
        paper_z=0.0019,
        clearance=s["clearance_m"],
        speed=s["writing_speed_m_s"],
        travel_speed=s["travel_speed_m_s"],
        columns=s["columns"],
        continuous_strokes=s.get("continuous_strokes", False),
        stroke_timing=s.get("stroke_timing", "quintic"),
    )


def character_stroke_offsets(poem: str, settings: dict[str, Any] | None = None) -> list[tuple[int, int]]:
    """Return inclusive global stroke-index ranges for each character."""
    s = load_settings() if settings is None else settings
    offsets = []
    total = 0
    for index, char in enumerate(poem):
        origin = (
            s["origin_xy_m"][0] + (index % s["columns"]) * (s["character_size_m"] + s["character_gap_m"]),
            s["origin_xy_m"][1] - (index // s["columns"]) * (s["character_size_m"] + s["character_gap_m"]),
        )
        char_path = make_path(
            char,
            origin_xy=origin,
            character_size=s["character_size_m"],
            gap=s["character_gap_m"],
            paper_z=0.0019,
            clearance=s["clearance_m"],
            speed=s["writing_speed_m_s"],
            travel_speed=s["travel_speed_m_s"],
            columns=1,
            continuous_strokes=s.get("continuous_strokes", False),
            stroke_timing=s.get("stroke_timing", "quintic"),
        )
        count = int(np.flatnonzero(np.diff(np.r_[False, char_path.pen_down].astype(int)) == 1).size)
        offsets.append((total, total + count))
        total += count
    return offsets


def proxy_character_scores(case_dir: Path, poem: str, settings: dict[str, Any] | None = None) -> list[int]:
    """Grade each character from the physical replay.

    The development proxy is intentionally strict:
    * 2: every stroke started and each stroke kept >= 70% contact frames.
    * 1: every stroke started and the character kept >= 50% contact frames.
    * 0: incomplete or unrecognisable in the physics trace.
    """
    case_dir = Path(case_dir)
    report_path = case_dir / "run/report.json"
    replay_path = case_dir / "run/replay.npz"
    if not report_path.exists() or not replay_path.exists():
        return [0] * len(poem)

    s = load_run_settings(case_dir)
    path = build_full_path(poem, s)
    starts = np.flatnonzero(np.diff(np.r_[False, path.pen_down].astype(int)) == 1)
    if len(starts) == 0:
        return [0] * len(poem)
    starts = np.r_[starts, len(path.xyz)]
    replay = np.load(replay_path, allow_pickle=False)
    achieved = int(replay["path_index"].max()) if "path_index" in replay else len(path.xyz) - 1
    indices = replay["path_index"]
    ink = replay["ink"]
    scores = []
    for first, end_exclusive in character_stroke_offsets(poem, s):
        fractions = []
        for stroke in range(first, min(end_exclusive, len(starts) - 1)):
            begin = int(starts[stroke])
            finish = int(min(starts[stroke + 1] - 1, achieved))
            if finish < begin:
                fractions.append(0.0)
                continue
            mask = (indices >= begin) & (indices <= finish)
            if not np.any(mask):
                fractions.append(0.0)
                continue
            # Ink row layout is [mark, contact_x, contact_y, contact_z, paper_N].
            # Column 4 is the measured paper normal force for that sample.
            fractions.append(float(np.mean(ink[mask, 4] > 0.01)))
        if not fractions or any(f <= 0.0 for f in fractions):
            scores.append(0)
        elif min(fractions) >= 0.7:
            scores.append(2)
        else:
            scores.append(1)
    return scores


def perfect_field_max(poem: str) -> int:
    """Reference field maximum: every character 2 plus all consecutive bonuses."""
    return final_raw_quality_points([2] * len(poem))


def evaluate_case(case_dir: Path, field_max_raw_quality_points: float | None = None) -> dict[str, Any]:
    """Judge a completed final case using the revised 2026 scoring contract."""
    case_dir = Path(case_dir)
    report = json.loads((case_dir / "run/report.json").read_text(encoding="utf-8"))
    settings = load_run_settings(case_dir)
    poem = str(report["recognized_poem"])
    scores = proxy_character_scores(case_dir, poem, settings)
    rules = load_competition_rules()
    maximum = perfect_field_max(poem) if field_max_raw_quality_points is None else float(field_max_raw_quality_points)
    process = report.get("process_evidence", {})
    # Legacy report names the hold/dip evidence differently; accept both layouts.
    if "penalty_points" in report:
        penalty_points = float(report["penalty_points"])
    else:
        # Legacy physical report: a brush drop is the only currently simulated
        # penalty. Stop reasons that end without a held brush incur 20 points.
        stopped_dropped = report.get("stop_reason") in ("grip_lost", "brush_dropped")
        penalty_points = 20.0 if stopped_dropped or report.get("pen_held_at_end") is False else 0.0
    score = score_final_round(
        touched_brush=bool(process.get("touched_brush", report.get("touched_brush_evidenced", report.get("touch_brush_evidenced", False)))),
        stable_hold_3s=bool(process.get("stable_hold_3s", report.get("hold_3s_passed", False))),
        dipped_ink=bool(process.get("dipped_ink", report.get("ink_loaded", False) and report.get("ink_withdrawn", False))),
        character_scores=scores,
        field_max_raw_quality_points=maximum,
        penalty_points=penalty_points,
        operation_mode_multiplier=1.0,
    )
    return {
        "poem": poem,
        "character_scores": scores,
        "characters_started": sum(1 for value in scores if value >= 1),
        "characters_grade_2": sum(1 for value in scores if value == 2),
        "characters_grade_1": sum(1 for value in scores if value == 1),
        "consecutive_bonus": final_consecutive_bonus(scores),
        "reference_field_max_raw_quality": maximum,
        "scoring": score,
        "official_human_judge_required": True,
    }

from __future__ import annotations

import json

import pytest

from fivefinger.competition_rules import (
    best_preliminary_attempt,
    final_consecutive_bonus,
    final_raw_quality_points,
    load_competition_rules,
    normalize_final_quality,
    preliminary_result_points,
    score_final_round,
    score_preliminary_attempt,
)
from fivefinger.paths import ROOT


def test_preliminary_result_boundaries() -> None:
    eye = 0.010
    assert preliminary_result_points(0.009, eye) == 360
    assert preliminary_result_points(0.014, eye) == 300
    assert preliminary_result_points(0.019, eye) == 240
    assert preliminary_result_points(0.024, eye) == 120
    assert preliminary_result_points(0.026, eye) == 0


def test_preliminary_attempt_score_and_invalid_ink() -> None:
    score = score_preliminary_attempt(
        touched_brush=True,
        stable_hold_3s=True,
        dipped_ink=True,
        ink_outer_radius_m=0.012,
        ink_radius_m=0.004,
        eye_radius_m=0.010,
    )
    assert score["process_points"] == 140
    assert score["result_points"] == 300
    assert score["raw_points"] == 440

    invalid = score_preliminary_attempt(
        touched_brush=True,
        stable_hold_3s=True,
        dipped_ink=True,
        ink_outer_radius_m=0.003,
        ink_radius_m=0.002,
        eye_radius_m=0.010,
    )
    assert invalid["ink_valid"] is False
    assert invalid["result_points"] == 0


def test_best_preliminary_attempt_uses_best_score() -> None:
    attempts = [
        {"attempt_id": "a", "raw_points": 300, "weighted_points": 300, "elapsed_s": 100},
        {"attempt_id": "b", "raw_points": 500, "weighted_points": 500, "elapsed_s": 170},
        {"attempt_id": "c", "raw_points": 500, "weighted_points": 500, "elapsed_s": 120},
    ]
    best = best_preliminary_attempt(attempts)
    assert best["best_attempt_id"] == "c"


def test_final_consecutive_bonus_does_not_reuse_characters() -> None:
    assert final_consecutive_bonus([1] * 20) == 20
    assert final_consecutive_bonus([1, 1, 1, 1, 0, 1, 1, 1, 1, 1]) == 5
    assert final_consecutive_bonus([2] * 20) == 20


def test_final_normalized_score() -> None:
    score = score_final_round(
        touched_brush=True,
        stable_hold_3s=True,
        dipped_ink=True,
        character_scores=[1] * 20,
        field_max_raw_quality_points=40,
    )
    assert score["process_points"] == 140
    assert score["raw_quality_points"] == 40
    assert score["normalized_quality_points"] == 1860.0
    assert score["weighted_points"] == 2000.0


def test_normalize_zero_denominator_is_zero() -> None:
    assert normalize_final_quality(0.0, 0.0) == 0.0


def test_final_scene_config_matches_revised_paper_contract() -> None:
    settings = json.loads((ROOT / "config/competition_scene_2026_final.json").read_text(encoding="utf-8"))
    assert settings["paper_layout"] == "four_a3_2x2_no_gap"
    assert len(settings["paper_sheets"]) == 4
    assert settings["round_limit_s"] == 300
    # Four A3 sheets (297x420 mm) cover the nominal 594x840 mm no-gap area.
    widths = [sheet["size_m"][0] for sheet in settings["paper_sheets"]]
    heights = [sheet["size_m"][1] for sheet in settings["paper_sheets"]]
    assert widths == [0.297, 0.297, 0.297, 0.297]
    assert sum(widths[:2]) == settings["paper_size_m"][0]
    assert sum(heights[::2]) == settings["paper_size_m"][1]

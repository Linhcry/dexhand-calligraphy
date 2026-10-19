from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "competition_20260911.json"
)


class CompetitionRuleError(ValueError):
    """Raised when the competition rule contract is missing or inconsistent."""


def load_competition_rules(path: str | Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    """Load and validate the current structured competition rule contract."""
    rule_path = Path(path)
    if not rule_path.is_file():
        raise CompetitionRuleError(f"Competition rule file not found: {rule_path}")

    try:
        payload = json.loads(rule_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompetitionRuleError(
            f"Invalid JSON in competition rule file {rule_path}: {exc}"
        ) from exc

    validate_competition_rules(payload)
    return payload


def validate_competition_rules(rules: dict[str, Any]) -> None:
    """Check the constraints that the autonomous runtime must never contradict."""
    required_sections = {
        "source",
        "selected_operation_mode",
        "compute",
        "hardware",
        "preliminary",
        "final",
        "penalties",
        "submission",
    }
    missing = sorted(required_sections.difference(rules))
    if missing:
        raise CompetitionRuleError(f"Missing rule sections: {', '.join(missing)}")

    mode = rules["selected_operation_mode"]
    if mode.get("name") != "full_autonomous":
        raise CompetitionRuleError("This project is configured for full_autonomous mode")
    if mode.get("human_command_during_round_allowed") is not False:
        raise CompetitionRuleError("Human commands must be disabled during a round")
    if mode.get("mode_switch_during_round_allowed") is not False:
        raise CompetitionRuleError("Mode switching must be disabled during a round")

    compute = rules["compute"]
    if compute.get("local_only") is not True:
        raise CompetitionRuleError("Competition computation must remain local")
    if compute.get("cloud_compute_allowed") is not False:
        raise CompetitionRuleError("Cloud computation must be disabled")
    if compute.get("remote_server_allowed") is not False:
        raise CompetitionRuleError("Remote servers must be disabled")

    preliminary = rules["preliminary"]
    if preliminary.get("attempts") != 3 or preliminary.get("seconds_per_attempt") != 180:
        raise CompetitionRuleError("Preliminary schedule must be 3 attempts of 180 seconds")

    final = rules["final"]
    paper = final.get("paper", {})
    if final.get("round_seconds") != 300:
        raise CompetitionRuleError("Final round must be limited to 300 seconds")
    if paper.get("sheet_count") != 4 or paper.get("layout") != "2x2_no_gap":
        raise CompetitionRuleError("Final scene must use four A3 sheets in a 2x2 layout")




def operation_mode_multiplier(rules: dict[str, Any]) -> float:
    """Return the configured round multiplier (1.0 full autonomous, 0.5 teleop)."""
    mode = rules.get("selected_operation_mode", {})
    if mode.get("name") == "full_autonomous":
        return 1.0
    if mode.get("name") == "teleoperation":
        return 0.5
    raise CompetitionRuleError(f"Unsupported operation mode: {mode.get('name')!r}")


def preliminary_result_points(ink_outer_radius_m: float, eye_radius_m: float) -> int:
    """Score one dot from the official 2026-09-11 boundary-circle table.

    ``ink_outer_radius_m`` is measured from the true eye centre to the farthest
    outer edge of the ink mark. A mark whose own radius is at most one third of
    the eye radius is invalid and scores zero.
    """
    if eye_radius_m <= 0 or ink_outer_radius_m <= 0:
        return 0
    ratio = ink_outer_radius_m / eye_radius_m
    if ratio <= 1.0:
        return 360
    if ratio <= 1.5:
        return 300
    if ratio <= 2.0:
        return 240
    if ratio <= 2.5:
        return 120
    return 0


def preliminary_process_points(*, touched_brush: bool, stable_hold_3s: bool, dipped_ink: bool) -> int:
    return sum(
        (
            20 if touched_brush else 0,
            40 if stable_hold_3s else 0,
            80 if dipped_ink else 0,
        )
    )


def score_preliminary_attempt(
    *,
    touched_brush: bool,
    stable_hold_3s: bool,
    dipped_ink: bool,
    ink_outer_radius_m: float,
    ink_radius_m: float,
    eye_radius_m: float,
    penalty_points: float = 0.0,
    operation_mode_multiplier: float = 1.0,
) -> dict[str, Any]:
    """One preliminary attempt under the revised rules.

    The ink radius itself must exceed ``eye_radius / 3``; otherwise the result
    component is zero even if a mark was left. Process points remain unchanged.
    """
    if eye_radius_m <= 0:
        raise ValueError("eye_radius_m must be positive")
    if ink_radius_m <= 0:
        raise ValueError("ink_radius_m must be positive")
    if not 0.0 <= penalty_points <= 500.0:
        raise ValueError("penalty_points is outside the plausible preliminary score range")
    process = preliminary_process_points(
        touched_brush=touched_brush,
        stable_hold_3s=stable_hold_3s,
        dipped_ink=dipped_ink,
    )
    valid_ink = ink_radius_m > eye_radius_m / 3.0
    result = preliminary_result_points(ink_outer_radius_m, eye_radius_m) if valid_ink else 0
    raw = process + result - penalty_points
    return {
        "process_points": process,
        "result_points": result,
        "penalty_points": float(penalty_points),
        "raw_points": raw,
        "weighted_points": round(raw * operation_mode_multiplier, 2),
        "operation_mode_multiplier": operation_mode_multiplier,
        "ink_valid": valid_ink,
        "boundary_ratio": ink_outer_radius_m / eye_radius_m,
    }


def best_preliminary_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the best scored attempt; ties use the shorter elapsed time."""
    if not attempts:
        return None
    scored = [a for a in attempts if isinstance(a, dict)]
    if not scored:
        return None
    best = max(
        scored,
        key=lambda item: (
            float(item.get("weighted_points", item.get("raw_points", 0.0))),
            -float(item.get("elapsed_s", 10**9)),
        ),
    )
    return {
        "best_attempt_id": best.get("attempt_id"),
        "best_weighted_points": best.get("weighted_points"),
        "best_raw_points": best.get("raw_points"),
        "best_elapsed_s": best.get("elapsed_s"),
        "rule": "preliminary total is the best of three scored attempts",
    }


def final_character_quality_points(character_scores: list[int]) -> int:
    """Sum the 0/1/2 character grades (the official judge panel is still required)."""
    if any(score not in (0, 1, 2) for score in character_scores):
        raise ValueError("Character scores must each be 0, 1 or 2")
    return sum(character_scores)


def final_consecutive_bonus(character_scores: list[int]) -> int:
    """Non-overlapping consecutive-five-character bonuses.

    A block qualifies when all five characters score at least 1. The official
    rule additionally requires the block to be consecutive poem content; callers
    must pass scores in poem order and may supply ``eligible`` when needed.
    """
    scores = list(character_scores)
    if any(score not in (0, 1, 2) for score in scores):
        raise ValueError("Character scores must each be 0, 1 or 2")
    bonus = 0
    index = 0
    while index + 5 <= len(scores):
        if all(score >= 1 for score in scores[index:index + 5]):
            bonus += 5
            index += 5
        else:
            index += 1
    return bonus


def final_raw_quality_points(character_scores: list[int]) -> int:
    return final_character_quality_points(character_scores) + final_consecutive_bonus(character_scores)


def normalize_final_quality(raw_quality_points: float, max_raw_quality_points: float) -> float:
    """Normalise quality to at most 1860 using the best valid team in the field."""
    raw = float(raw_quality_points)
    maximum = float(max_raw_quality_points)
    if raw < 0 or maximum < 0:
        raise ValueError("Quality points cannot be negative")
    if maximum == 0:
        return 0.0
    return min(raw / maximum, 1.0) * 1860.0


def score_final_round(
    *,
    touched_brush: bool,
    stable_hold_3s: bool,
    dipped_ink: bool,
    character_scores: list[int],
    field_max_raw_quality_points: float | None = None,
    penalty_points: float = 0.0,
    operation_mode_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Final-round score. If the field maximum is unknown, use the local raw score."""
    process = preliminary_process_points(
        touched_brush=touched_brush,
        stable_hold_3s=stable_hold_3s,
        dipped_ink=dipped_ink,
    )
    raw_quality = final_raw_quality_points(character_scores)
    maximum = (
        float(field_max_raw_quality_points)
        if field_max_raw_quality_points is not None
        else float(raw_quality)
    )
    normalized_quality = normalize_final_quality(raw_quality, maximum)
    raw = process + normalized_quality - penalty_points
    return {
        "process_points": process,
        "raw_quality_points": raw_quality,
        "normalized_quality_points": round(normalized_quality, 2),
        "penalty_points": float(penalty_points),
        "raw_points": raw,
        "weighted_points": round(raw * operation_mode_multiplier, 2),
        "operation_mode_multiplier": operation_mode_multiplier,
    }


def round_limit_seconds(rules: dict[str, Any], stage: str) -> int:
    if stage == "preliminary":
        return int(rules["preliminary"]["seconds_per_attempt"])
    if stage == "final":
        return int(rules["final"]["round_seconds"])
    raise ValueError(f"Unknown competition stage: {stage}")


def assert_runtime_is_legal_for_round(rules: dict[str, Any], stage: str, elapsed_s: float) -> None:
    """Central guard used by both 2026 task runners before accepting a result."""
    if stage not in ("preliminary", "final"):
        raise ValueError(f"Unknown competition stage: {stage}")
    if not 0.0 <= elapsed_s <= round_limit_seconds(rules, stage):
        raise CompetitionRuleError(
            f"{stage} result timestamp {elapsed_s:.3f}s is outside the official round limit"
        )

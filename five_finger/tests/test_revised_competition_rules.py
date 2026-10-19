from __future__ import annotations

import copy

import pytest

from fivefinger.competition_rules import (
    CompetitionRuleError,
    load_competition_rules,
    validate_competition_rules,
)


def test_revised_rules_load_with_full_autonomous_local_runtime() -> None:
    rules = load_competition_rules()

    assert rules["selected_operation_mode"]["name"] == "full_autonomous"
    assert rules["compute"]["local_only"] is True
    assert rules["preliminary"]["attempts"] == 3
    assert rules["final"]["paper"]["sheet_count"] == 4


@pytest.mark.parametrize(
    ("section", "field", "invalid_value"),
    [
        ("selected_operation_mode", "human_command_during_round_allowed", True),
        ("compute", "cloud_compute_allowed", True),
        ("compute", "remote_server_allowed", True),
        ("preliminary", "attempts", 2),
        ("final", "round_seconds", 301),
    ],
)
def test_revised_rules_reject_conflicting_runtime_constraints(
    section: str,
    field: str,
    invalid_value: object,
) -> None:
    rules = copy.deepcopy(load_competition_rules())
    rules[section][field] = invalid_value

    with pytest.raises(CompetitionRuleError):
        validate_competition_rules(rules)


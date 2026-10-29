import math

import pytest

from asha_vani.schemas import ExtractedVisit, ValidationSeverity
from asha_vani.validation import BP_GROUP, LIMITS, validate_visit


@pytest.mark.parametrize("field,limits", LIMITS.items())
def test_hard_range_boundaries_are_accepted(field, limits):
    for boundary in limits.hard:
        visit = ExtractedVisit(**{field: boundary})

        result = validate_visit(visit)

        assert not result.has_errors
        assert getattr(result.values, field) == boundary


@pytest.mark.parametrize("field,limits", LIMITS.items())
def test_values_outside_hard_range_are_rejected(field, limits):
    minimum, maximum = limits.hard
    for value in (minimum - 1, maximum + 1):
        visit = ExtractedVisit(**{field: value})

        result = validate_visit(visit)

        assert result.has_errors
        assert getattr(result.values, field) is None
        assert any(
            issue.field == field
            and issue.code == "VALUE_OUTSIDE_HARD_LIMIT"
            and issue.severity is ValidationSeverity.ERROR
            for issue in result.issues
        )


@pytest.mark.parametrize("field,limits", LIMITS.items())
def test_soft_range_boundaries_do_not_warn(field, limits):
    for boundary in limits.soft:
        visit = ExtractedVisit(**{field: boundary})

        result = validate_visit(visit)

        assert not result.needs_confirmation
        assert getattr(result.values, field) == boundary


@pytest.mark.parametrize("field,limits", LIMITS.items())
def test_values_inside_hard_but_outside_soft_range_need_confirmation(field, limits):
    hard_minimum, hard_maximum = limits.hard
    soft_minimum, soft_maximum = limits.soft
    values = (soft_minimum - 1, soft_maximum + 1)

    for value in values:
        assert hard_minimum <= value <= hard_maximum
        visit = ExtractedVisit(**{field: value})

        result = validate_visit(visit)

        assert not result.has_errors
        assert result.needs_confirmation
        assert getattr(result.values, field) == value


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_hemoglobin_is_rejected(value):
    result = validate_visit(ExtractedVisit(hb_g_dl=value))

    assert result.has_errors
    assert result.values.hb_g_dl is None
    assert result.issues[0].code == "NON_FINITE_VALUE"


@pytest.mark.parametrize("systolic,diastolic", [(120, 120), (110, 120)])
def test_inconsistent_blood_pressure_clears_both_values(systolic, diastolic):
    visit = ExtractedVisit(bp_systolic=systolic, bp_diastolic=diastolic)

    result = validate_visit(visit)

    assert result.has_errors
    assert result.values.bp_systolic is None
    assert result.values.bp_diastolic is None
    assert any(
        issue.field == BP_GROUP
        and issue.code == "SYSTOLIC_NOT_GREATER_THAN_DIASTOLIC"
        for issue in result.issues
    )


def test_hard_invalid_bp_clears_only_that_reading_and_skips_pair_check():
    visit = ExtractedVisit(bp_systolic=300, bp_diastolic=95)

    result = validate_visit(visit)

    assert result.values.bp_systolic is None
    assert result.values.bp_diastolic == 95
    assert len(result.issues) == 1
    assert result.issues[0].code == "VALUE_OUTSIDE_HARD_LIMIT"


def test_missing_values_are_left_for_conversation_flow():
    visit = ExtractedVisit()

    result = validate_visit(visit)

    assert result.values == visit
    assert result.issues == []
    assert not result.has_errors
    assert not result.needs_confirmation


def test_validation_does_not_mutate_input_and_warning_keeps_value():
    visit = ExtractedVisit(gestational_weeks=3)

    result = validate_visit(visit)

    assert visit.gestational_weeks == 3
    assert result.values.gestational_weeks == 3
    assert result.needs_confirmation
    assert not result.has_errors

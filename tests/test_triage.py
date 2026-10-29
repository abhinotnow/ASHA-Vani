import pytest

from asha_vani.schemas import (
    ExtractedVisit,
    Presence,
    SymptomCode,
    SymptomReport,
    TriageLevel,
)
from asha_vani.triage import evaluate_triage
from asha_vani.validation import validate_visit


def symptom(code: SymptomCode, presence: Presence = Presence.PRESENT) -> SymptomReport:
    return SymptomReport(code=code, presence=presence)


@pytest.mark.parametrize(
    "visit",
    [
        ExtractedVisit(bp_systolic=160),
        ExtractedVisit(bp_diastolic=110),
    ],
)
def test_r1_severe_bp_is_critical(visit):
    result = evaluate_triage(visit)

    assert result.level is TriageLevel.CRITICAL
    assert "R1" in result.rules_fired


def test_r2_elevated_bp_with_severe_feature_is_critical():
    visit = ExtractedVisit(
        bp_systolic=140,
        bp_diastolic=90,
        symptoms=[symptom(SymptomCode.SEVERE_HEADACHE)],
    )

    result = evaluate_triage(visit)

    assert result.level is TriageLevel.CRITICAL
    assert result.rules_fired == ["R2"]


@pytest.mark.parametrize(
    "symptoms",
    [
        [],
        [symptom(SymptomCode.SWELLING)],
    ],
)
def test_r3_elevated_bp_without_severe_feature_is_urgent(symptoms):
    result = evaluate_triage(ExtractedVisit(bp_systolic=140, symptoms=symptoms))

    assert result.level is TriageLevel.URGENT
    assert result.rules_fired == ["R3"]


@pytest.mark.parametrize("weeks", [20, 24, 40])
def test_r4_bleeding_from_20_weeks_is_critical(weeks):
    visit = ExtractedVisit(
        gestational_weeks=weeks,
        symptoms=[symptom(SymptomCode.VAGINAL_BLEEDING)],
    )

    result = evaluate_triage(visit)

    assert result.level is TriageLevel.CRITICAL
    assert "R4" in result.rules_fired


def test_r4_does_not_fire_below_20_weeks_under_current_draft_rule():
    visit = ExtractedVisit(
        gestational_weeks=19,
        symptoms=[symptom(SymptomCode.VAGINAL_BLEEDING)],
    )

    result = evaluate_triage(visit)

    assert result.level is TriageLevel.ROUTINE
    assert "R4" not in result.rules_fired


@pytest.mark.parametrize(
    "hb,expected_level,expected_rules",
    [
        (7.0, TriageLevel.ROUTINE, []),
        (6.9, TriageLevel.URGENT, ["R5a"]),
        (5.0, TriageLevel.URGENT, ["R5a"]),
        (4.9, TriageLevel.CRITICAL, ["R5a", "R5b"]),
    ],
)
def test_r5_anemia_boundaries(hb, expected_level, expected_rules):
    result = evaluate_triage(ExtractedVisit(hb_g_dl=hb))

    assert result.level is expected_level
    assert result.rules_fired == expected_rules


@pytest.mark.parametrize(
    "code",
    [SymptomCode.CONVULSION, SymptomCode.UNCONSCIOUS],
)
def test_r6_convulsion_or_unconsciousness_is_critical_without_bp(code):
    result = evaluate_triage(ExtractedVisit(symptoms=[symptom(code)]))

    assert result.level is TriageLevel.CRITICAL
    assert "R6" in result.rules_fired


@pytest.mark.parametrize(
    "code",
    [SymptomCode.LABOUR_PAINS, SymptomCode.LEAKING_FLUID],
)
def test_r7_preterm_labor_or_leaking_before_37_weeks_is_critical(code):
    result = evaluate_triage(
        ExtractedVisit(gestational_weeks=36, symptoms=[symptom(code)])
    )

    assert result.level is TriageLevel.CRITICAL
    assert "R7" in result.rules_fired


@pytest.mark.parametrize("weeks", [37, 38, 40])
def test_r7_does_not_fire_at_or_after_37_weeks(weeks):
    result = evaluate_triage(
        ExtractedVisit(
            gestational_weeks=weeks,
            symptoms=[symptom(SymptomCode.LABOUR_PAINS)],
        )
    )

    assert "R7" not in result.rules_fired


@pytest.mark.parametrize("weeks", [28, 32, 40])
def test_r8_reduced_fetal_movement_from_28_weeks_is_urgent(weeks):
    result = evaluate_triage(
        ExtractedVisit(
            gestational_weeks=weeks,
            symptoms=[symptom(SymptomCode.REDUCED_FETAL_MOVEMENT)],
        )
    )

    assert result.level is TriageLevel.URGENT
    assert "R8" in result.rules_fired


def test_unknown_and_absent_symptoms_do_not_fire_rules():
    visit = ExtractedVisit(
        bp_systolic=140,
        symptoms=[
            symptom(SymptomCode.SEVERE_HEADACHE, Presence.UNKNOWN),
            symptom(SymptomCode.SWELLING, Presence.ABSENT),
        ],
    )

    result = evaluate_triage(visit)

    assert result.level is TriageLevel.URGENT
    assert result.rules_fired == ["R3"]


def test_critical_rule_wins_and_all_matching_rules_are_kept():
    result = evaluate_triage(ExtractedVisit(bp_systolic=160))

    assert result.level is TriageLevel.CRITICAL
    assert result.rules_fired == ["R1", "R3"]


def test_rejected_bp_does_not_hide_a_critical_symptom():
    raw_visit = ExtractedVisit(
        bp_systolic=300,
        bp_diastolic=95,
        symptoms=[symptom(SymptomCode.CONVULSION)],
    )
    validation = validate_visit(raw_visit)

    result = evaluate_triage(validation.values)

    assert validation.values.bp_systolic is None
    assert validation.values.bp_diastolic == 95
    assert result.level is TriageLevel.CRITICAL
    assert result.rules_fired == ["R2", "R6"]
    assert "R1" not in result.rules_fired

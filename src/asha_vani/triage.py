"""Deterministic screening rules for ASHA-Vani.

These are draft project rules and require clinician review. This module does
not diagnose; it reports a screening level and the rules that matched.
"""

from asha_vani.schemas import (
    ExtractedVisit,
    SymptomCode,
    TriageLevel,
    TriageResult,
)


_LEVEL_PRIORITY = {
    TriageLevel.ROUTINE: 0,
    TriageLevel.URGENT: 1,
    TriageLevel.CRITICAL: 2,
}

_SEVERE_BP_SYMPTOMS = {
    SymptomCode.SEVERE_HEADACHE,
    SymptomCode.BLURRED_VISION,
    SymptomCode.EPIGASTRIC_PAIN,
    SymptomCode.CONVULSION,
}


def evaluate_triage(visit: ExtractedVisit) -> TriageResult:
    """Evaluate draft rules against a validated/sanitized visit.

    Call this with ``ValidationResult.values`` so rejected measurements have
    already been cleared. Symptoms can still trigger a rule when a measurement
    is missing or was rejected. The highest matching level is returned, and
    every matching rule ID is retained for explanation.
    """

    fired: list[tuple[str, TriageLevel]] = []
    symptoms = visit.present_symptoms
    systolic = visit.bp_systolic
    diastolic = visit.bp_diastolic
    hb = visit.hb_g_dl
    weeks = visit.gestational_weeks

    # A screening BP threshold is met if either component reaches its value.
    bp_at_least_140_90 = (systolic is not None and systolic >= 140) or (
        diastolic is not None and diastolic >= 90
    )

    # R1: severe-range blood pressure.
    if (systolic is not None and systolic >= 160) or (
        diastolic is not None and diastolic >= 110
    ):
        fired.append(("R1", TriageLevel.CRITICAL))

    # R2: elevated blood pressure with a severe feature.
    if bp_at_least_140_90 and symptoms & _SEVERE_BP_SYMPTOMS:
        fired.append(("R2", TriageLevel.CRITICAL))

    # R3: elevated blood pressure without a severe feature; swelling alone
    # does not count as a severe feature.
    if bp_at_least_140_90 and not (symptoms & _SEVERE_BP_SYMPTOMS):
        fired.append(("R3", TriageLevel.URGENT))

    # R4: vaginal bleeding at or beyond 20 gestational weeks.
    if (
        weeks is not None
        and weeks >= 20
        and SymptomCode.VAGINAL_BLEEDING in symptoms
    ):
        fired.append(("R4", TriageLevel.CRITICAL))

    # R5a/R5b: anemia screening thresholds. Both can fire below 5; the
    # highest level wins while preserving both rule IDs for the case record.
    if hb is not None and hb < 7:
        fired.append(("R5a", TriageLevel.URGENT))
    if hb is not None and hb < 5:
        fired.append(("R5b", TriageLevel.CRITICAL))

    # R6: convulsion or unconsciousness is critical regardless of BP.
    if symptoms & {SymptomCode.CONVULSION, SymptomCode.UNCONSCIOUS}:
        fired.append(("R6", TriageLevel.CRITICAL))

    # R7: labour pains or leaking fluid before 37 weeks.
    preterm_labor_symptoms = {
        SymptomCode.LABOUR_PAINS,
        SymptomCode.LEAKING_FLUID,
    }
    if (
        weeks is not None
        and weeks < 37
        and symptoms & preterm_labor_symptoms
    ):
        fired.append(("R7", TriageLevel.CRITICAL))

    # R8: reduced fetal movement from 28 weeks.
    if (
        weeks is not None
        and weeks >= 28
        and SymptomCode.REDUCED_FETAL_MOVEMENT in symptoms
    ):
        fired.append(("R8", TriageLevel.URGENT))

    if not fired:
        return TriageResult(level=TriageLevel.ROUTINE)

    highest_level = max(
        (level for _, level in fired), key=_LEVEL_PRIORITY.__getitem__
    )
    return TriageResult(
        level=highest_level,
        rules_fired=[rule_id for rule_id, _ in fired],
    )

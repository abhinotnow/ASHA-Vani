"""Basic range and consistency checks for extracted visit values.

The thresholds below are draft project values from the architecture plan.
They require clinician review before being described as clinically approved.
"""

import math
from dataclasses import dataclass

from asha_vani.schemas import (
    ExtractedVisit,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)


# The graph treats this issue field as "ask for blood pressure again" (both
# numbers), rather than as the name of one model field.
BP_GROUP = "blood_pressure"
_BP_FIELDS = {"bp_systolic", "bp_diastolic"}


@dataclass(frozen=True)
class FieldLimits:
    """Label and hard/soft inclusive bounds for one numeric field."""

    label: str
    hard: tuple[float, float]
    soft: tuple[float, float]

    def __post_init__(self) -> None:
        hard_min, hard_max = self.hard
        soft_min, soft_max = self.soft
        if not hard_min <= soft_min <= soft_max <= hard_max:
            raise ValueError(f"{self.label}: soft range must sit inside hard range")


# One source of truth for each field's label and draft bounds.
LIMITS: dict[str, FieldLimits] = {
    "gestational_weeks": FieldLimits(
        "Gestational weeks", hard=(1, 44), soft=(4, 42)
    ),
    "bp_systolic": FieldLimits(
        "Systolic blood pressure", hard=(60, 260), soft=(80, 240)
    ),
    "bp_diastolic": FieldLimits(
        "Diastolic blood pressure", hard=(30, 160), soft=(40, 140)
    ),
    "hb_g_dl": FieldLimits("Hemoglobin (g/dL)", hard=(2.0, 20.0), soft=(3.0, 18.0)),
}


def _outside(value: float, bounds: tuple[float, float]) -> bool:
    """Return whether a value falls outside an inclusive (min, max) range."""

    minimum, maximum = bounds
    return value < minimum or value > maximum


def validate_visit(visit: ExtractedVisit) -> ValidationResult:
    """Report range errors, confirmation warnings, and BP inconsistency.

    The returned values are a copy with unusable numbers set to None. Missing
    values are left alone; the conversation graph decides what to ask next.
    The input visit is never modified.
    """

    issues: list[ValidationIssue] = []
    unusable: set[str] = set()

    for field, limits in LIMITS.items():
        value = getattr(visit, field)
        if value is None:
            continue

        # NaN and infinity compare false against range bounds, so reject them first.
        if not math.isfinite(value):
            unusable.add(field)
            issues.append(
                ValidationIssue(
                    field=field,
                    code="NON_FINITE_VALUE",
                    message=f"{limits.label} must be a finite number; received {value}.",
                    severity=ValidationSeverity.ERROR,
                )
            )
            continue

        if _outside(value, limits.hard):
            unusable.add(field)
            issues.append(
                ValidationIssue(
                    field=field,
                    code="VALUE_OUTSIDE_HARD_LIMIT",
                    message=(
                        f"{limits.label} must be between {limits.hard[0]} and "
                        f"{limits.hard[1]}; received {value}."
                    ),
                    severity=ValidationSeverity.ERROR,
                )
            )
            continue

        if _outside(value, limits.soft):
            issues.append(
                ValidationIssue(
                    field=field,
                    code="VALUE_OUTSIDE_SOFT_LIMIT",
                    message=(
                        f"{limits.label} is outside the usual confirmation range "
                        f"({limits.soft[0]} to {limits.soft[1]}); read it back."
                    ),
                    severity=ValidationSeverity.WARNING,
                )
            )

    systolic = visit.bp_systolic
    diastolic = visit.bp_diastolic
    bp_has_unusable_value = bool(_BP_FIELDS & unusable)
    if (
        systolic is not None
        and diastolic is not None
        and not bp_has_unusable_value
        and systolic <= diastolic
    ):
        # Either number may be wrong, so neither is trusted.
        unusable.update(_BP_FIELDS)
        issues.append(
            ValidationIssue(
                field=BP_GROUP,
                code="SYSTOLIC_NOT_GREATER_THAN_DIASTOLIC",
                message=(
                    "Systolic blood pressure must be greater than diastolic "
                    f"blood pressure; received {systolic}/{diastolic}."
                ),
                severity=ValidationSeverity.ERROR,
            )
        )

    # model_copy preserves the original object and only clears rejected numbers.
    usable_visit = visit.model_copy(update={field: None for field in unusable})
    return ValidationResult(values=usable_visit, issues=issues)

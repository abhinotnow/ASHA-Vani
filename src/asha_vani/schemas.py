"""Shared Pydantic models for ASHA-Vani's core logic.

These describe internal data passed between extraction, validation, and
triage. They are not yet database tables or frontend API contracts.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field



class ProjectModel(BaseModel):
    """Base model that rejects fields we have not explicitly defined."""

    model_config = ConfigDict(extra="forbid")


class TriageLevel(str, Enum):
    """Screening level returned by the deterministic rule engine."""

    ROUTINE = "ROUTINE"
    URGENT = "URGENT"
    CRITICAL = "CRITICAL"


class ValidationSeverity(str, Enum):
    """Whether a validation finding blocks triage or needs confirmation."""

    ERROR = "ERROR"
    WARNING = "WARNING"


class Presence(str, Enum):
    """Whether a symptom was reported, absent, or not established."""

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNKNOWN = "UNKNOWN"


class SymptomCode(str, Enum):
    """Symptoms used by the current draft triage rules."""

    SEVERE_HEADACHE = "SEVERE_HEADACHE"
    BLURRED_VISION = "BLURRED_VISION"
    EPIGASTRIC_PAIN = "EPIGASTRIC_PAIN"
    CONVULSION = "CONVULSION"
    UNCONSCIOUS = "UNCONSCIOUS"
    VAGINAL_BLEEDING = "VAGINAL_BLEEDING"
    LABOUR_PAINS = "LABOUR_PAINS"
    LEAKING_FLUID = "LEAKING_FLUID"
    REDUCED_FETAL_MOVEMENT = "REDUCED_FETAL_MOVEMENT"
    SWELLING = "SWELLING"


class SymptomReport(ProjectModel):
    """One symptom and the ASHA's reported status for it."""

    code: SymptomCode
    presence: Presence
    phrase: str | None = None


class PatientReference(ProjectModel):
    """Spoken patient details passed to roster matching, not a database row."""

    full_name: str | None = None
    spouse_or_guardian_name: str | None = None
    age_years: int | None = None
    village: str | None = None
    tola: str | None = None


class ExtractedVisit(ProjectModel):
    """Candidate values extracted from one completed utterance.

    Fields are optional because an ASHA may share visit details over several
    turns. This model checks data shape and type; clinical range checks belong
    in the validation module so unclear or implausible values can trigger a
    repeat. Omitted symptoms mean no report for this utterance, not ABSENT.
    """

    gestational_weeks: int | None = None
    bp_systolic: int | None = None
    bp_diastolic: int | None = None
    hb_g_dl: float | None = None
    symptoms: list[SymptomReport] = Field(default_factory=list)

    @property
    def present_symptoms(self) -> set[SymptomCode]:
        """Explicitly present symptoms; UNKNOWN and ABSENT never count."""

        return {
            symptom.code
            for symptom in self.symptoms
            if symptom.presence is Presence.PRESENT
        }


class ValidationIssue(ProjectModel):
    """One field-level finding from validation."""

    field: str
    code: str
    message: str
    severity: ValidationSeverity


class ValidationResult(ProjectModel):
    """Validated candidate values and any findings to handle before triage."""

    values: ExtractedVisit
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """True when at least one finding blocks use of these values."""

        return any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)

    @property
    def needs_confirmation(self) -> bool:
        """True when a warning should be read back to the ASHA."""

        return any(issue.severity is ValidationSeverity.WARNING for issue in self.issues)


class TriageResult(ProjectModel):
    """Screening level and IDs of rules that produced it."""

    level: TriageLevel
    rules_fired: list[str] = Field(default_factory=list)

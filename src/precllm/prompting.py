from __future__ import annotations

from dataclasses import dataclass

from .constants import SHOT_TO_EXAMPLE_COUNT
from .enums import Phenotype


@dataclass(frozen=True)
class PromptExample:
    """One supervised in-context example."""

    text: str
    label: int


@dataclass(frozen=True)
class PromptTemplate:
    """Prompt template for one phenotype."""

    task_name: str
    user_query: str
    examples: tuple[PromptExample, ...]


_METASTASIS_EXAMPLES: tuple[PromptExample, ...] = (
    PromptExample("Innumerable hepatic and pulmonary metastases.", 1),
    PromptExample("New diagnosis of metastatic cancer.", 1),
    PromptExample("Metastatic disease in liver and lumbar spine.", 1),
    PromptExample("There is a metastatic lesion involving T5 vertebral body.", 1),
    PromptExample("He now has metastatic disease.", 1),
    PromptExample("Metastatic laryngeal SCC to L1 vertebra and liver.", 1),
    PromptExample("No metastatic disease.", 2),
    PromptExample("No evidence of metastatic disease in the brain.", 2),
    PromptExample("Staging CT chest was negative for metastases.", 2),
    PromptExample("No definitive metastatic disease in the chest.", 2),
    PromptExample("There was no evidence of distant metastases.", 2),
    PromptExample("No metastatic carcinoma identified (0/1 node).", 2),
    PromptExample("Patient reports abdominal pain for several days.", 3),
    PromptExample("No gross rectal bleeding noted in emergency department.", 3),
    PromptExample("No history of head injury or loss of consciousness.", 3),
    PromptExample("Progressing through chemotherapy schedule this week.", 3),
    PromptExample("Ten systems reviewed, otherwise within normal limits.", 3),
    PromptExample("No deep venous thrombosis in lower extremity veins.", 3),
)

_INSULIN_EXAMPLES: tuple[PromptExample, ...] = (
    PromptExample("Started insulin glargine nightly for persistent hyperglycemia.", 1),
    PromptExample("Currently on basal-bolus insulin regimen.", 1),
    PromptExample("Insulin infusion initiated in ICU.", 1),
    PromptExample("Home medications include insulin lispro before meals.", 1),
    PromptExample("Patient is insulin-dependent for diabetes management.", 1),
    PromptExample("Required correctional insulin during admission.", 1),
    PromptExample("Diabetes is controlled on metformin only, no insulin.", 2),
    PromptExample("Patient denies insulin use at home.", 2),
    PromptExample("No insulin ordered in medication administration record.", 2),
    PromptExample("Endocrinology note: continue oral agents, avoid insulin.", 2),
    PromptExample("No evidence of insulin therapy in this encounter.", 2),
    PromptExample("Type 2 diabetes without current insulin treatment.", 2),
    PromptExample("Glucose elevated; treatment plan to be determined.", 3),
    PromptExample("Diabetes history documented, medication list unavailable.", 3),
    PromptExample("Nutrition counseling note without medication details.", 3),
    PromptExample("Patient transferred before medication reconciliation.", 3),
    PromptExample("Hyperglycemia discussed but no therapy recorded.", 3),
    PromptExample("Follow-up plan mentions endocrine referral only.", 3),
)

_HYPERTENSION_EXAMPLES: tuple[PromptExample, ...] = (
    PromptExample("Known hypertension on lisinopril and amlodipine.", 1),
    PromptExample("History of HTN documented in problem list.", 1),
    PromptExample("Hypertensive urgency treated with intravenous medication.", 1),
    PromptExample("Long-standing high blood pressure requiring two agents.", 1),
    PromptExample("Primary diagnosis includes essential hypertension.", 1),
    PromptExample("Blood pressure remains elevated, consistent with hypertension.", 1),
    PromptExample("No history of hypertension.", 2),
    PromptExample("Normotensive throughout admission.", 2),
    PromptExample("Patient denies high blood pressure diagnosis.", 2),
    PromptExample("No antihypertensive medications on home list.", 2),
    PromptExample("Assessment: no evidence of chronic hypertension.", 2),
    PromptExample("Blood pressure normal, HTN ruled out.", 2),
    PromptExample("Single elevated reading likely due to pain.", 3),
    PromptExample("Blood pressure fluctuating; diagnosis not established.", 3),
    PromptExample("Cardiology follow-up planned for blood pressure evaluation.", 3),
    PromptExample("Vitals incomplete in transferred records.", 3),
    PromptExample("Hypertension mentioned in family history only.", 3),
    PromptExample("Outpatient records unavailable for confirmation.", 3),
)

PROMPT_LIBRARY: dict[Phenotype, PromptTemplate] = {
    Phenotype.METASTASIS: PromptTemplate(
        task_name="metastasis status",
        user_query="Analyze the following clinical notes to determine metastasis status:",
        examples=_METASTASIS_EXAMPLES,
    ),
    Phenotype.INSULIN: PromptTemplate(
        task_name="insulin use status",
        user_query="Analyze the following clinical notes to determine insulin use status:",
        examples=_INSULIN_EXAMPLES,
    ),
    Phenotype.HYPERTENSION: PromptTemplate(
        task_name="hypertension status",
        user_query="Analyze the following clinical notes to determine hypertension status:",
        examples=_HYPERTENSION_EXAMPLES,
    ),
}


def _task_header(task_name: str) -> str:
    return (
        "Task: Classify the presence of "
        f"{task_name} from patient clinical notes. Respond only with numeric codes:\n"
        "- (1) Yes: Evidence clearly supports presence.\n"
        "- (2) No: Evidence clearly supports absence.\n"
        "- (3) Unknown: Insufficient or conflicting evidence.\n\n"
        "Instructions:\n"
        "1. Do not provide explanations.\n"
        "2. Use only the provided note text.\n"
        "3. Output exactly one code: (1), (2), or (3).\n"
    )


def build_prompt(phenotype: Phenotype, shot: int, note_text: str) -> str:
    """Build a model prompt using phenotype and shot configuration."""

    template = PROMPT_LIBRARY[phenotype]
    requested_examples = SHOT_TO_EXAMPLE_COUNT[shot]
    selected_examples = template.examples[:requested_examples]

    lines: list[str] = ["<start_of_turn>user", _task_header(template.task_name)]
    if selected_examples:
        lines.append("Examples:")
        for index, example in enumerate(selected_examples, start=1):
            lines.append(f'- Example {index}: "{example.text}" --> ({example.label})')

    lines.append("")
    lines.append(f"{template.user_query}\n{note_text}<end_of_turn>")
    lines.append("<start_of_turn>model")
    return "\n".join(lines)


def prompt_metadata(phenotype: Phenotype, shot: int) -> dict[str, object]:
    """Return metadata about the selected prompt template."""

    template = PROMPT_LIBRARY[phenotype]
    example_count = SHOT_TO_EXAMPLE_COUNT[shot]
    return {
        "phenotype": phenotype.value,
        "shot": shot,
        "example_count": example_count,
        "task_name": template.task_name,
    }

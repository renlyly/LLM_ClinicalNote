# Subject-level summary

This folder packages a minimal subject-level workflow that starts from SUM CSVs (no note IDs) and produces two sensitivity plots: model sensitivity and system sensitivity.

## Contents

- `data/sum_subject/system_sensitivity/` and `data/sum_subject/model_sensitivity/`: input SUM CSVs.
- `scripts/aggregate_subject_summaries.py`: aggregates SUM CSVs into consolidated summaries.
- `scripts/plot_subject_model_sensitivity.R`: plots model sensitivity.
- `scripts/plot_subject_system_sensitivity.R`: plots system sensitivity.

## Run

From the repo root:

```bash
python Context_learning/Summary/scripts/aggregate_subject_summaries.py
Rscript Context_learning/Summary/scripts/plot_subject_model_sensitivity.R
Rscript Context_learning/Summary/scripts/plot_subject_system_sensitivity.R
```

Outputs are written to `Context_learning/Summary/output/`:

- `model_sensitive_3model_private_subject.pdf`
- `sys_sensitive_3model_private_subject.pdf`

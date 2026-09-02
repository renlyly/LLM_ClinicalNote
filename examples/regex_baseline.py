#!/usr/bin/env python3
"""Apply the 24-term NegEx baseline to regex-retrieved text.
The script labels each input and combines its labels by subject-level majority vote."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


NEGATION_TERMS = [
    "absence of",
    "declined",
    "denied",
    "denies",
    "denying",
    "did not exhibit",
    "doubt",
    "negative for",
    "no",
    "no cause of",
    "no complaints of",
    "no evidence of",
    "no sign of",
    "no signs of",
    "not",
    "not demonstrate",
    "patient was not",
    "ruled out",
    "rules out",
    "unlikely",
    "versus",
    "without",
    "without indication of",
    "without sign of",
]

parts = []
for term in sorted(NEGATION_TERMS, key=len, reverse=True):
    left = r"(?<!\w)" if term[0].isalnum() else ""
    right = r"(?!\w)" if term[-1].isalnum() else ""
    parts.append(left + re.escape(term) + right)
NEGATION_PATTERN = re.compile("|".join(parts), re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--labeled-inputs-csv", required=True)
    parser.add_argument("--subject-labels-csv", required=True)
    parser.add_argument("--id-column", default="subject_id")
    parser.add_argument("--text-column", default="text")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    labeled_path = Path(args.labeled_inputs_csv)
    subject_path = Path(args.subject_labels_csv)
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    subject_path.parent.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(lambda: {"positive": 0, "negative": 0})

    with input_path.open(newline="", encoding="utf-8") as source, labeled_path.open(
        "w", newline="", encoding="utf-8"
    ) as output:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("The input CSV must have a header row.")

        missing = [
            column
            for column in (args.id_column, args.text_column)
            if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError("Missing CSV column(s): " + ", ".join(missing))

        writer = csv.DictWriter(
            output, fieldnames=reader.fieldnames + ["regex_baseline_label"]
        )
        writer.writeheader()

        for row_number, row in enumerate(reader, start=2):
            subject_id = row[args.id_column].strip()
            if not subject_id:
                raise ValueError(
                    f"Missing {args.id_column} in input CSV row {row_number}."
                )

            if NEGATION_PATTERN.search(row[args.text_column] or ""):
                label = "negative"
            else:
                label = "positive"
            counts[subject_id][label] += 1
            row["regex_baseline_label"] = label
            writer.writerow(row)

    with subject_path.open("w", newline="", encoding="utf-8") as output:
        fields = [
            args.id_column,
            "positive_inputs",
            "negative_inputs",
            "regex_subject_label",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()

        for subject_id in sorted(counts):
            positive = counts[subject_id]["positive"]
            negative = counts[subject_id]["negative"]
            if positive > negative:
                subject_label = "positive"
            elif negative > positive:
                subject_label = "negative"
            else:
                subject_label = "undecided"

            writer.writerow(
                {
                    args.id_column: subject_id,
                    "positive_inputs": positive,
                    "negative_inputs": negative,
                    "regex_subject_label": subject_label,
                }
            )


if __name__ == "__main__":
    main()

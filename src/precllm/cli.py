from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .constants import DEFAULT_MODEL_PATHS, SHOT_OPTIONS
from .enums import InferenceBackend, ModelName, Phenotype, PreprocessMethod
from .exceptions import PrecLLMError
from .logging_utils import configure_logging
from .models import RunConfig
from .pipeline import run_pipeline
from .prompting import build_prompt, prompt_metadata


def _parse_model(value: str) -> ModelName:
    try:
        return ModelName(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in ModelName)
        raise ValueError(f"Invalid model '{value}'. Allowed values: {choices}") from exc


def _parse_phenotype(value: str) -> Phenotype:
    try:
        return Phenotype(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in Phenotype)
        raise ValueError(f"Invalid phenotype '{value}'. Allowed values: {choices}") from exc


def _parse_preprocess(value: str) -> PreprocessMethod:
    try:
        return PreprocessMethod(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in PreprocessMethod)
        raise ValueError(f"Invalid preprocess method '{value}'. Allowed values: {choices}") from exc


def _parse_backend(value: str) -> InferenceBackend:
    try:
        return InferenceBackend(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in InferenceBackend)
        raise ValueError(f"Invalid inference backend '{value}'. Allowed values: {choices}") from exc


def _parse_id_columns(args: argparse.Namespace) -> tuple[str, ...]:
    if args.id_columns:
        parsed = tuple(part.strip() for part in args.id_columns.split(",") if part.strip())
        if parsed:
            return parsed
    return (args.id_column,)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="precllm",
        description=(
            "Enterprise-grade paper workflow package for configurable clinical note "
            "phenotyping experiments."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run preprocess + prediction workflow.")
    run_parser.add_argument("--input-csv", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--model-path", default=None)
    run_parser.add_argument("--phenotype", required=True)
    run_parser.add_argument("--preprocess", required=True)
    run_parser.add_argument("--shot", required=True, type=int, choices=SHOT_OPTIONS)
    run_parser.add_argument("--seed", default=100, type=int)
    run_parser.add_argument("--rag-top-k", default=3, type=int)
    run_parser.add_argument("--text-column", default="text")
    run_parser.add_argument("--id-column", default="note_id")
    run_parser.add_argument(
        "--id-columns",
        default="",
        help="Comma-separated id columns. Example: subject_id,hadm_id",
    )
    run_parser.add_argument(
        "--inference-backend",
        default=InferenceBackend.RULE.value,
        help="rule | transformers",
    )
    run_parser.add_argument("--max-new-tokens", default=8, type=int)
    run_parser.add_argument("--temperature", default=0.1, type=float)
    run_parser.add_argument("--do-sample", action="store_true")
    run_parser.add_argument("--log-level", default="INFO")
    run_parser.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("catalog", help="Print supported configuration values.")

    prompt_parser = subparsers.add_parser(
        "prompt", help="Render the exact prompt template used for one phenotype and shot setting."
    )
    prompt_parser.add_argument("--phenotype", required=True)
    prompt_parser.add_argument("--shot", required=True, type=int, choices=SHOT_OPTIONS)
    prompt_parser.add_argument(
        "--note-text",
        default="Example clinical note text.",
        help="Sample note text used when rendering the prompt preview.",
    )

    return parser


def _catalog_payload() -> dict[str, object]:
    return {
        "models": [item.value for item in ModelName],
        "model_defaults": {key.value: value for key, value in DEFAULT_MODEL_PATHS.items()},
        "phenotypes": [item.value for item in Phenotype],
        "preprocess_methods": [item.value for item in PreprocessMethod],
        "inference_backends": [item.value for item in InferenceBackend],
        "shot_values": list(SHOT_OPTIONS),
        "prompt_module": "src/precllm/prompting.py",
    }


def _run_command(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)

    model = _parse_model(args.model)
    phenotype = _parse_phenotype(args.phenotype)
    preprocess = _parse_preprocess(args.preprocess)
    backend = _parse_backend(args.inference_backend)
    id_columns = _parse_id_columns(args)

    config = RunConfig(
        model=model,
        phenotype=phenotype,
        preprocess=preprocess,
        shot=args.shot,
        seed=args.seed,
        rag_top_k=args.rag_top_k,
        text_column=args.text_column,
        id_columns=id_columns,
        inference_backend=backend,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        dry_run=args.dry_run,
    )

    if config.dry_run:
        preview_meta = prompt_metadata(config.phenotype, config.shot)
        print("Dry run enabled. No files were generated.")
        print(json.dumps(_catalog_payload(), indent=2))
        print(
            json.dumps(
                {
                    "input_csv": str(args.input_csv),
                    "output_dir": str(args.output_dir),
                    "config": {
                        "model": config.model.value,
                        "model_path": config.model_path or DEFAULT_MODEL_PATHS[config.model],
                        "phenotype": config.phenotype.value,
                        "preprocess": config.preprocess.value,
                        "inference_backend": config.inference_backend.value,
                        "shot": config.shot,
                        "seed": config.seed,
                        "rag_top_k": config.rag_top_k,
                        "text_column": config.text_column,
                        "id_columns": list(config.id_columns),
                        "max_new_tokens": config.max_new_tokens,
                        "temperature": config.temperature,
                        "do_sample": config.do_sample,
                    },
                    "prompt_metadata": preview_meta,
                },
                indent=2,
            )
        )
        return 0

    summary = run_pipeline(args.input_csv, args.output_dir, config)
    print(
        json.dumps(
            {
                "total_notes": summary.total_notes,
                "yes": summary.yes_count,
                "no": summary.no_count,
                "unknown": summary.unknown_count,
                "output_dir": str(summary.output_dir),
            },
            indent=2,
        )
    )
    return 0


def _prompt_command(args: argparse.Namespace) -> int:
    phenotype = _parse_phenotype(args.phenotype)
    payload = {
        "prompt_metadata": prompt_metadata(phenotype, args.shot),
        "prompt_preview": build_prompt(phenotype, args.shot, args.note_text),
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "catalog":
            print(json.dumps(_catalog_payload(), indent=2))
            return 0
        if args.command == "prompt":
            return _prompt_command(args)
        if args.command == "run":
            return _run_command(args)

        raise ValueError(f"Unsupported command '{args.command}'.")
    except (ValueError, PrecLLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .constants import DEFAULT_MODEL_PATHS
from .enums import ModelName
from .exceptions import PipelineExecutionError


class LLMBackend(Protocol):
    """Runtime protocol for model generation backends."""

    def generate(self, prompt: str) -> str:
        """Generate raw model output for a prompt."""


@dataclass
class TransformersGenerationConfig:
    """Generation hyperparameters for Hugging Face inference."""

    max_new_tokens: int = 8
    temperature: float = 0.1
    do_sample: bool = False


class TransformersBackend:
    """Transformers backend with lazy imports and robust runtime errors."""

    def __init__(
        self,
        model: ModelName,
        model_path: str | None,
        generation: TransformersGenerationConfig,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise PipelineExecutionError(
                "Transformers backend requires the 'transformers' package. "
                "Install with: pip install transformers torch"
            ) from exc

        resolved_path = model_path or DEFAULT_MODEL_PATHS[model]
        self._model_path = resolved_path
        self._generation = generation

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(resolved_path, use_fast=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                resolved_path,
                device_map="auto",
            )
        except Exception as exc:  # pragma: no cover
            raise PipelineExecutionError(
                f"Failed to initialize transformers model at '{resolved_path}'."
            ) from exc

    @property
    def model_path(self) -> str:
        """Resolved model identifier/path in use."""

        return self._model_path

    def generate(self, prompt: str) -> str:
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            input_length = inputs["input_ids"].shape[1]
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._generation.max_new_tokens,
                do_sample=self._generation.do_sample,
                temperature=self._generation.temperature,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            return self._tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
        except Exception as exc:  # pragma: no cover
            raise PipelineExecutionError("Model generation failed during transformers inference.") from exc

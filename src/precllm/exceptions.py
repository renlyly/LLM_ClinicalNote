class PrecLLMError(Exception):
    """Base exception for package-specific errors."""


class ConfigValidationError(PrecLLMError):
    """Raised when user configuration fails validation."""


class DataValidationError(PrecLLMError):
    """Raised when input data schema or contents are invalid."""


class PipelineExecutionError(PrecLLMError):
    """Raised when a pipeline stage cannot complete successfully."""

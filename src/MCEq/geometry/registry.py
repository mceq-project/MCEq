"""Compatibility exports of the environment-model registry."""

from MCEq.environment.registry import (  # noqa: F401
    DENSITY_MODELS,
    available_models,
    build,
    density_model_class,
)

__all__ = [
    "DENSITY_MODELS",
    "available_models",
    "build",
    "density_model_class",
]

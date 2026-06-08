"""SOC model roster. Importing this package registers every model in the
registry (classical + deep) so they can be built by string name.
"""
from .base import BaseModel, build_model, list_models, register, MODEL_REGISTRY  # noqa
from . import classical  # noqa: F401  (registers `linear`)
from . import torch_models  # noqa: F401  (registers the deep roster)

__all__ = ["BaseModel", "build_model", "list_models", "register", "MODEL_REGISTRY"]

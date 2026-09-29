"""
Checkpoint helpers for multiVIB models.

Functions
---------
save_checkpoint             Save model weights + constructor config (+ optimizer).
load_checkpoint             Load a checkpoint file into an existing model.
setup_model_from_checkpoint Rebuild a model from a checkpoint file alone.
"""

import inspect
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn

from . import models as _models

# Model classes that can be reconstructed from a checkpoint, keyed by class name.
MODEL_REGISTRY: Dict[str, type] = {
    name: cls
    for name, cls in vars(_models).items()
    if inspect.isclass(cls) and issubclass(cls, nn.Module)
}


def _infer_config(model: nn.Module) -> Dict[str, Any]:
    """
    Reconstruct the constructor kwargs of a multiVIB model from its live
    sub-modules, so that ``cls(**config)`` rebuilds an architecture whose
    ``state_dict`` shapes match the saved one.

    Init-only arguments that leave no trace in the architecture (``relation``,
    ``relations``) are omitted; translator masks live in the ``state_dict`` as
    registered buffers and are restored by ``load_state_dict``.
    """
    config: Dict[str, Any] = {}
    signature = inspect.signature(type(model).__init__)
    for name in signature.parameters:
        if hasattr(model, name):
            value = getattr(model, name)
            config[name] = list(value) if isinstance(value, tuple) else value

    # Arguments not stored as attributes, recovered from module shapes.
    # The projecter is a plain linear layer, so no config to recover there.
    net = getattr(getattr(model, "classifier", None), "net", None)
    if net is not None and "n_class" in signature.parameters:
        config["n_class"] = net[-1].out_features
    return config


def save_checkpoint(model: nn.Module, path: Union[str, Path]) -> Path:
    """
    Save a model checkpoint that is sufficient to rebuild the model later
    with :func:`setup_model_from_checkpoint`.

    Args:
        model: A multiVIB model (any class from :mod:`multivib.models`).
        path:  Destination file, e.g. ``"checkpoints/multivib_epoch100.pt"``.
               Parent directories are created if missing.

    Returns:
        The path the checkpoint was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_class": type(model).__name__,
        "model_config": _infer_config(model),
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)
    return path


def _torch_load(path: Union[str, Path], map_location) -> Dict[str, Any]:
    """torch.load with weights_only=False (checkpoints hold plain-python config)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only argument
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint into an existing model.

    Args:
        path:         Checkpoint file written by :func:`save_checkpoint`.
        model:        Model instance to load the weights into.
        map_location: Device to map tensors to (``"cpu"``, ``"cuda:0"``, ...).
        strict:       Passed to ``load_state_dict``.

    Returns:
        The full checkpoint dict.
    """
    checkpoint = _torch_load(path, map_location)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return checkpoint


def setup_model_from_checkpoint(
    path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    eval_mode: bool = True,
    strict: bool = True,
    **config_overrides: Any,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Rebuild a multiVIB model from a checkpoint file alone: reconstruct the
    architecture from the saved class name and config, then load the weights.

    Args:
        path:              Checkpoint file written by :func:`save_checkpoint`.
        map_location:      Device to map tensors to; the model is also moved there.
        eval_mode:         If ``True`` (default) the model is set to ``.eval()``;
                           set ``False`` when resuming training.
        strict:            Passed to ``load_state_dict``; set ``False`` to load
                           a checkpoint across model variants (e.g. a different
                           projecter architecture).
        **config_overrides: Constructor kwargs to override the saved config
                           (only safe for args that don't change parameter
                           shapes, e.g. ``joint=False``).

    Returns:
        ``(model, checkpoint)`` — the rebuilt model and the full checkpoint dict.

    Example::

        model, ckpt = setup_model_from_checkpoint("checkpoints/run1.pt")
        out = model(x_a, x_b, batcha, batchb)
    """
    checkpoint = _torch_load(path, map_location)

    class_name = checkpoint["model_class"]
    if class_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model class {class_name!r} in checkpoint. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )

    config = {**checkpoint["model_config"], **config_overrides}
    model = MODEL_REGISTRY[class_name](**config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.to(map_location)
    if eval_mode:
        model.eval()
    return model, checkpoint

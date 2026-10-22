"""Import optional dependencies with an install hint naming their extra."""

from __future__ import annotations

import importlib
import types


def import_optional(module: str, *, extra: str, feature: str) -> types.ModuleType:
    """Import and return an optional module, or raise naming its extra.

    Parameters
    ----------
    module : str
        Importable module name, e.g. ``"librosa"``.
    extra : str
        The audiospylt extra that installs it, e.g. ``"librosa"``.
    feature : str
        What needs the module, used as the message subject.

    Returns
    -------
    types.ModuleType
        The imported module.

    Raises
    ------
    ImportError
        ``"{feature} requires {module}; install audiospylt[{extra}]"`` when
        ``module`` is not installed, with the ModuleNotFoundError as its cause.
        Other import errors of an installed module propagate unchanged.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        # A dependency failing inside an installed module is not a missing extra.
        if (error.name or "").partition(".")[0] != module.partition(".")[0]:
            raise
        raise ImportError(
            f"{feature} requires {module}; install audiospylt[{extra}]"
        ) from error

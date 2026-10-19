"""Imports of the packages that the optional extras install."""

import importlib
from types import ModuleType


def missing_extra_message(package: str, extra: str) -> str:
    """Return the error message for package when it is not installed.

    The message names the pip command that installs extra.
    """
    return (
        f"{package} is not installed. Install the {extra} extra: "
        f"pip install 'timeseries-zarr[{extra}]'"
    )


def require(package: str, extra: str) -> ModuleType:
    """Import package and return the module.

    Raises ImportError with the message of missing_extra_message, and with
    name set to package, if package is not installed. If a module that
    package imports is missing, its ModuleNotFoundError propagates.
    """
    try:
        return importlib.import_module(package)
    except ModuleNotFoundError as error:
        if error.name != package:
            raise
        raise ImportError(
            missing_extra_message(package, extra), name=package
        ) from error

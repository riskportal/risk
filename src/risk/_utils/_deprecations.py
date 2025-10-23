"""
risk/_utils/_deprecations
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict, Optional

from .._log import logger


def map_deprecated_kwarg(
    *,
    kwargs: Dict[str, Any],
    old: str,
    new: str,
    message: Optional[str] = None,
) -> Optional[Any]:
    """
    If ``old`` is present in ``kwargs``, emit a deprecation warning and map it to ``new`` in-place.
    Returns the value from the deprecated key, or None if the deprecated key was not present.

    Args:
        kwargs (Dict[str, Any]): The keyword arguments dictionary to check and modify.
        old (str): The deprecated keyword argument name.
        new (str): The new keyword argument name to map to.
        message (Optional[str]): Custom deprecation warning message.

    Returns:
        Optional[Any]: The value from the deprecated key, or None if not present.
    """
    if old not in kwargs:
        return None

    # Consume the deprecated value first
    deprecated_value = kwargs.pop(old)

    # Otherwise, map over to the new key and warn about deprecation
    logger.warning(message or f"\nWARNING: '{old}' is deprecated; use '{new}' instead.\n")
    kwargs[new] = deprecated_value

    return deprecated_value

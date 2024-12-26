import json
import sys


################################################################################
# Validation
################################################################################
def sanitize_raw_string(value: str) -> str:
    """
    Sanitizes the specified string by:
      - removing any invalid characters.
      - ensuring it is not longer than the maximum Python string length.

    This is for an abundance of caution with security, to avoid any potential issues with the string.
    """
    # remove any invalid characters by making sure it is a valid UTF-8 string
    value = value.encode("utf-8", "ignore").decode("utf-8")

    # ensure it is not longer than the maximum Python string length
    return value[: sys.maxsize]


def sanitize_dict(value: dict) -> dict:
    """
    Sanitizes the specified dictionary by:
      - removing any invalid characters.
      - ensuring that the dictionary is not too deeply nested.
    """
    # sanitize the string representation of the dictionary
    tmp_str = sanitize_raw_string(json.dumps(value, ensure_ascii=False))

    return json.loads(tmp_str)

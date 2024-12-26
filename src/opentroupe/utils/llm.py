import copy
import json
import re
from collections.abc import Callable
from typing import Any

from .log import logger
from .rendering import break_text_at_length


################################################################################
# Model output utilities
################################################################################
def extract_json(text: str) -> dict:
    """
    Extracts a JSON object from a string, ignoring: any text before the first
    opening curly brace; and any Markdown opening (```json) or closing(```) tags.
    """
    try:
        # remove any text before the first opening curly or square braces, using regex. Leave the braces.
        text = re.sub(r"^.*?({|\[)", r"\1", text, flags=re.DOTALL)

        # remove any trailing text after the LAST closing curly or square braces, using regex. Leave the braces.
        text = re.sub(r"(}|\])(?!.*(\]|\})).*$", r"\1", text, flags=re.DOTALL)

        # remove invalid escape sequences, which show up sometimes
        # replace \' with just '
        text = re.sub("\\'", "'", text)  # re.sub(r'\\\'', r"'", text)

        # remove new lines, tabs, etc.
        text = text.replace("\n", "").replace("\t", "").replace("\r", "")

        # return the parsed JSON object
        return json.loads(text)

    except Exception:  # noqa: BLE001
        return {}


################################################################################
# Model control utilities
################################################################################


def repeat_on_error(retries: int, exceptions: list) -> Callable:
    """
    Decorator that repeats the specified function call if an exception among those specified occurs,
    up to the specified number of retries. If that number of retries is exceeded, the
    exception is raised. If no exception occurs, the function returns normally.

    Args:
        retries (int): The number of retries to attempt.
        exceptions (list): The list of exception classes to catch.

    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> None:
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions) as e:  # noqa: PERF203
                    logger.debug(f"Exception occurred: {e}")
                    if i == retries - 1:
                        raise
                    logger.debug(f"Retrying ({i+1}/{retries})...")
                    continue
            return None

        return wrapper

    return decorator


################################################################################
# Truncation
################################################################################


def truncate_actions_or_stimuli(
    list_of_actions_or_stimuli: list[dict],
    max_content_length: int,
) -> list[dict]:
    """
    Truncates the content of actions or stimuli at the specified maximum length. Does not modify the original list.

    Args:
        list_of_actions_or_stimuli (Collection[dict]): The list of actions or stimuli to truncate.
        max_content_length (int): The maximum length of the content.

    Returns:
        Collection[str]: The truncated list of actions or stimuli. It is a new list, not a reference to the original list,
        to avoid unexpected side effects.

    """
    cloned_list = copy.deepcopy(list_of_actions_or_stimuli)

    for element in cloned_list:
        # the external wrapper of the LLM message: {'role': ..., 'content': ...}
        if "content" in element:
            msg_content = element["content"]

            # now the actual action or stimulus content

            # has action, stimuli or stimulus as key?
            if "action" in msg_content:
                # is content there?
                if "content" in msg_content["action"]:
                    msg_content["action"]["content"] = break_text_at_length(
                        msg_content["action"]["content"],
                        max_content_length,
                    )
            elif "stimulus" in msg_content:
                # is content there?
                if "content" in msg_content["stimulus"]:
                    msg_content["stimulus"]["content"] = break_text_at_length(
                        msg_content["stimulus"]["content"],
                        max_content_length,
                    )
            elif "stimuli" in msg_content:
                # for each element in the list
                for stimulus in msg_content["stimuli"]:
                    # is content there?
                    if "content" in stimulus:
                        stimulus["content"] = break_text_at_length(
                            stimulus["content"],
                            max_content_length,
                        )

    return cloned_list

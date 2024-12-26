import json
from datetime import datetime


################################################################################
# Rendering and markup
################################################################################
def break_text_at_length(text: str | dict, max_length: int | None = None) -> str:
    """
    Breaks the text (or JSON) at the specified length, inserting a "(...)" string at the break point.
    If the maximum length is `None`, the content is returned as is.
    """
    if isinstance(text, dict):
        text = json.dumps(text, indent=4)

    if max_length is None or len(text) <= max_length:
        return text
    return text[:max_length] + " (...)"


def pretty_datetime(dt: datetime) -> str:
    """Returns a pretty string representation of the specified datetime object."""
    return dt.strftime("%Y-%m-%d %H:%M")


class RichTextStyle:
    STIMULUS_CONVERSATION_STYLE = "bold italic cyan1"
    STIMULUS_THOUGHT_STYLE = "dim italic cyan1"
    STIMULUS_DEFAULT_STYLE = "italic"
    ACTION_DONE_STYLE = "grey82"
    ACTION_TALK_STYLE = "bold green3"
    ACTION_THINK_STYLE = "green"
    ACTION_DEFAULT_STYLE = "purple"

    @classmethod
    def get_style_for(cls, kind: str, event_type: str) -> str | None:  # noqa: PLR0911
        if kind in ("stimulus", "stimuli"):
            if event_type == "CONVERSATION":
                return cls.STIMULUS_CONVERSATION_STYLE
            if event_type == "THOUGHT":
                return cls.STIMULUS_THOUGHT_STYLE
            return cls.STIMULUS_DEFAULT_STYLE

        if kind == "action":
            if event_type == "DONE":
                return cls.ACTION_DONE_STYLE
            if event_type == "TALK":
                return cls.ACTION_TALK_STYLE
            if event_type == "THINK":
                return cls.ACTION_THINK_STYLE
            return cls.ACTION_DEFAULT_STYLE
        return None

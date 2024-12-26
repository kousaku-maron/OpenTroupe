from typing import Any


################################################################################
# Other
################################################################################
def name_or_empty(named_entity: Any) -> str:
    """Returns the name of the specified agent or environment, or an empty string if the agent is None."""
    if named_entity is None:
        return ""
    return named_entity.name

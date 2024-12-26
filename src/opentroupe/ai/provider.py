from typing import Any, TypedDict

###########################################################################
# Default parameter values
###########################################################################


class DefaultParameters(TypedDict):
    max_attempts: int
    waiting_time: int
    exponential_backoff_factor: int


default: DefaultParameters = {
    "max_attempts": 5,
    "waiting_time": 1,
    "exponential_backoff_factor": 5,
}


class NextMessage(TypedDict):
    role: str
    content: str


class Provider:
    def send_message(
        self,
        messages: list,
        max_attempts: int = default["max_attempts"],
        waiting_time: int = default["waiting_time"],
        exponential_backoff_factor: int = default["exponential_backoff_factor"],
        response_format: Any | None = None,
    ) -> NextMessage | None:
        """Sends a message to the LLM API and returns the response."""
        raise NotImplementedError("Subclasses must implement this method.")

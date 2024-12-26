###########################################################################
# Exceptions
###########################################################################
class InvalidRequestError(Exception):
    """Exception raised when the request to the OpenAI API is invalid."""


class NonTerminalError(Exception):
    """Exception raised when an unspecified error occurs but we know we can retry."""

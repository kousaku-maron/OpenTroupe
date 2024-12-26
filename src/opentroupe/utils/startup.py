from collections.abc import Callable
from typing import Any


def post_init(cls: Any) -> Callable[[Any], None]:
    """
    Decorator to enforce a post-initialization method call in a class, if it has one.
    The method must be named `_post_init`.
    """
    original_init = cls.__init__

    def new_init(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN001
        original_init(self, *args, **kwargs)
        if hasattr(self, "_post_init"):
            self._post_init()

    cls.__init__ = new_init
    return cls

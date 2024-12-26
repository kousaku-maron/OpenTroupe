from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from opentroupe.agent import OpenPerson

#######################################################################################################################
# Mental faculties
#######################################################################################################################


class OpenMentalFaculty:
    """Represents a mental faculty of an agent. Mental faculties are the cognitive abilities that an agent has."""

    def __init__(self, name: str, requires_faculties: list | None = None) -> None:
        """
        Initializes the mental faculty.

        Args:
            name (str): The name of the mental faculty.
            requires_faculties (list, optional): A list of mental faculties that this faculty requires to function properly.

        """
        self.name = name

        if requires_faculties is None:
            self.requires_faculties = []
        else:
            self.requires_faculties = requires_faculties

    def process_action(self, agent: OpenPerson, action: dict) -> bool:
        """
        Processes an action related to this faculty.

        Args:
            agent (OpenPerson): The agent.
            action (dict): The action to process.

        Returns:
            bool: True if the action was successfully processed, False otherwise.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def actions_definitions_prompt(self) -> str:
        """Returns the prompt for defining a actions related to this faculty."""
        raise NotImplementedError("Subclasses must implement this method.")

    def actions_constraints_prompt(self) -> str:
        """Returns the prompt for defining constraints on actions related to this faculty."""
        raise NotImplementedError("Subclasses must implement this method.")


#######################################################################################################################
# Memory mechanisms
#######################################################################################################################


class OpenMemory(OpenMentalFaculty):
    """Base class for different types of memory."""

    def _preprocess_value_for_storage(self, value: Any) -> Any:
        """Preprocesses a value before storing it in memory."""
        # by default, we don't preprocess the value
        return value

    def _store(self, value: Any) -> None:
        """Stores a value in memory."""
        raise NotImplementedError("Subclasses must implement this method.")

    def store(self, value: dict) -> None:
        """Stores a value in memory."""
        self._store(self._preprocess_value_for_storage(value))

    def retrieve(
        self,
        first_n: int,
        last_n: int,
        include_omission_info: bool = True,
    ) -> list:
        """
        Retrieves the first n and/or last n values from memory. If n is None, all values are retrieved.

        Args:
            first_n (int): The number of first values to retrieve.
            last_n (int): The number of last values to retrieve.
            include_omission_info (bool): Whether to include an information message when some values are omitted.

        Returns:
            list: The retrieved values.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve_recent(self) -> list:
        """Retrieves the n most recent values from memory."""
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve_all(self) -> list:
        """Retrieves all values from memory."""
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve_relevant(self, relevance_target: str, top_k: int = 20) -> list:
        """Retrieves all values from memory that are relevant to a given target."""
        raise NotImplementedError("Subclasses must implement this method.")


class EpisodicMemory(OpenMemory):
    """
    Provides episodic memory capabilities to an agent. Cognitively, episodic memory is the ability to remember specific events,
    or episodes, in the past. This class provides a simple implementation of episodic memory, where the agent can store and retrieve
    messages from memory.

    Subclasses of this class can be used to provide different memory implementations.
    """

    MEMORY_BLOCK_OMISSION_INFO: ClassVar = {
        "role": "assistant",
        "content": "Info: there were other messages here, but they were omitted for brevity.",
        "simulation_timestamp": None,
    }

    def __init__(
        self,
        fixed_prefix_length: int = 100,
        lookback_length: int = 100,
    ) -> None:
        """
        Initializes the memory.

        Args:
            fixed_prefix_length (int): The fixed prefix length. Defaults to 20.
            lookback_length (int): The lookback length. Defaults to 20.

        """
        self.fixed_prefix_length = fixed_prefix_length
        self.lookback_length = lookback_length

        self.memory: list = []

    def _store(self, value: Any) -> None:
        """Stores a value in memory."""
        self.memory.append(value)

    def count(self) -> int:
        """Returns the number of values in memory."""
        return len(self.memory)

    def retrieve(
        self,
        first_n: int,
        last_n: int,
        include_omission_info: bool = True,
    ) -> list:
        """
        Retrieves the first n and/or last n values from memory. If n is None, all values are retrieved.

        Args:
            first_n (int): The number of first values to retrieve.
            last_n (int): The number of last values to retrieve.
            include_omission_info (bool): Whether to include an information message when some values are omitted.

        Returns:
            list: The retrieved values.

        """
        omisssion_info = (
            [EpisodicMemory.MEMORY_BLOCK_OMISSION_INFO] if include_omission_info else []
        )

        # use the other methods in the class to implement
        if first_n is not None and last_n is not None:
            return (
                self.retrieve_first(first_n, include_omission_info=False)
                + omisssion_info
                + self.retrieve_last(last_n, include_omission_info=False)
            )
        if first_n is not None:
            return self.retrieve_first(first_n, include_omission_info)
        if last_n is not None:
            return self.retrieve_last(last_n, include_omission_info)
        return self.retrieve_all()

    def retrieve_recent(self, include_omission_info: bool = True) -> list:
        """Retrieves the n most recent values from memory."""
        omisssion_info = (
            [EpisodicMemory.MEMORY_BLOCK_OMISSION_INFO] if include_omission_info else []
        )

        # compute fixed prefix
        fixed_prefix = self.memory[: self.fixed_prefix_length] + omisssion_info

        # how many lookback values remain?
        remaining_lookback = min(
            len(self.memory) - len(fixed_prefix),
            self.lookback_length,
        )

        # compute the remaining lookback values and return the concatenation
        if remaining_lookback <= 0:
            return fixed_prefix
        return fixed_prefix + self.memory[-remaining_lookback:]

    def retrieve_all(self) -> list:
        """Retrieves all values from memory."""
        return copy.copy(self.memory)

    def retrieve_relevant(self, relevance_target: str, top_k: int = 20) -> list:
        """Retrieves top-k values from memory that are most relevant to a given target."""
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve_first(self, n: int, include_omission_info: bool = True) -> list:
        """Retrieves the first n values from memory."""
        omisssion_info = (
            [EpisodicMemory.MEMORY_BLOCK_OMISSION_INFO] if include_omission_info else []
        )

        return self.memory[:n] + omisssion_info

    def retrieve_last(self, n: int, include_omission_info: bool = True) -> list:
        """Retrieves the last n values from memory."""
        omisssion_info = (
            [EpisodicMemory.MEMORY_BLOCK_OMISSION_INFO] if include_omission_info else []
        )

        return omisssion_info + self.memory[-n:]


# TODO: Implement the following classes
class SemanticMemory(OpenMemory):
    """
    Semantic memory is the memory of meanings, understandings, and other concept-based knowledge unrelated to specific experiences.
    It is not ordered temporally, and it is not about remembering specific events or episodes. This class provides a simple implementation
    of semantic memory, where the agent can store and retrieve semantic information.
    """

    def __init__(self) -> None:
        pass

    # TODO: Implement the methods below
    def retrieve_relevant(self, relevance_target: str, top_k=20) -> list:  # noqa: ANN001, ARG002
        """Retrieves all values from memory that are relevant to a given target."""
        return []

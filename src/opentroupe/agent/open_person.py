from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict

import chevron
from pydantic import BaseModel
from rich import print  # noqa: A004

from opentroupe import utils
from opentroupe.ai import OpenaiProvider
from opentroupe.utils import post_init, repeat_on_error

from .memory import EpisodicMemory, SemanticMemory
from .prompts import open_person_prompt_template

if TYPE_CHECKING:
    from opentroupe.environment import OpenWorld

###########################################################################
# Default parameter values
###########################################################################


class DefaultParameters(TypedDict):
    max_content_display_length: int


default: DefaultParameters = {
    "max_content_display_length": 1024,
}


###########################################################################
# Data structures to enforce output format during LLM API call.
###########################################################################
class Action(BaseModel):
    type: str
    content: str
    target: str


class CognitiveState(BaseModel):
    goals: str
    attention: str
    emotions: str


class CognitiveActionModel(BaseModel):
    action: Action
    cognitive_state: CognitiveState


@post_init
class OpenPerson:
    """A simulated person in the OpenTroupe universe."""

    # The maximum number of actions that an agent is allowed to perform before DONE.
    # This prevents the agent from acting without ever stopping.
    MAX_ACTIONS_BEFORE_DONE = 15

    PP_TEXT_WIDTH = 100

    serializable_attributes: ClassVar[list[str]] = [
        "name",
        "episodic_memory",
        "semantic_memory",
        "_mental_faculties",
        "_configuration",
    ]

    # A dict of all agents instantiated so far.
    all_agents: ClassVar[dict[str, object]] = {}  # name -> agent

    # The communication style for all agents: "simplified" or "full".
    communication_style: str = "simplified"

    # Whether to display the communication or not. True is for interactive applications,
    # when we want to see simulation outputs as they are produced.
    communication_display: bool = True

    def __init__(
        self,
        name: str | None = None,
        episodic_memory: EpisodicMemory | None = None,
        semantic_memory: SemanticMemory | None = None,
        mental_faculties: list | None = None,
    ) -> None:
        """
        Create a OpenPerson.

        Args:
            name (str): The name of the OpenPerson. Either this or spec_path must be specified.
            episodic_memory (EpisodicMemory, optional): The memory implementation to use. Defaults to EpisodicMemory().
            semantic_memory (SemanticMemory, optional): The memory implementation to use. Defaults to SemanticMemory().
            mental_faculties (list, optional): A list of mental faculties to add to the agent. Defaults to None.

        """
        # NOTE: default values will be given in the _post_init method, as that's shared by
        #       direct initialization as well as via deserialization.
        if episodic_memory is not None:
            self.episodic_memory = episodic_memory

        if semantic_memory is not None:
            self.semantic_memory = semantic_memory

        # Mental faculties
        if mental_faculties is not None:
            self._mental_faculties = mental_faculties

        if name is None:
            raise ValueError("A OpenPerson must have a name.")
        self.name = name

        # @post_init makes sure that _post_init is called after __init__

    def _post_init(self, **kwargs) -> None:  # noqa: ANN003, ARG002
        """
        _post_init will run after __init__, since the class has the @post_init decorator.
        It is convenient to separate some of the initialization processes to make deserialize easier.
        """
        ############################################################
        # Default values
        ############################################################

        self.current_messages: list = []

        # the current environment in which the agent is acting
        self.environment: OpenWorld | None = None

        # The list of actions that this agent has performed so far, but which have not been
        # consumed by the environment yet.
        self._actions_buffer: list[object] = []

        # The list of agents that this agent can currently interact with.
        # This can change over time, as agents move around the world.
        self._accessible_agents: list[OpenPerson] = []

        # the buffer of communications that have been displayed so far, used for
        # saving these communications to another output form later (e.g., caching)
        self._displayed_communications_buffer: list[object] = []

        if not hasattr(self, "episodic_memory"):
            self.episodic_memory = EpisodicMemory()

        if not hasattr(self, "semantic_memory"):
            self.semantic_memory = SemanticMemory()

        if not hasattr(self, "_mental_faculties"):
            # This default value MUST NOT be in the method signature, otherwise it will be shared across all instances.
            self._mental_faculties = []

        if not hasattr(self, "_configuration"):
            self._configuration: dict = {
                "name": self.name,
                "age": None,
                "nationality": None,
                "country_of_residence": None,
                "occupation": None,
                "routines": [],
                "occupation_description": None,
                "personality_traits": [],
                "professional_interests": [],
                "personal_interests": [],
                "skills": [],
                "relationships": [],
                "current_datetime": None,
                "current_location": None,
                "current_context": [],
                "current_attention": None,
                "current_goals": [],
                "current_emotions": "Currently you feel calm and friendly.",
                "current_memory_context": None,
                "currently_accessible_agents": [],  # [{"agent": agent_1, "relation": "My friend"}, {"agent": agent_2, "relation": "My colleague"}, ...]
            }

        if not hasattr(self, "_extended_agent_summary"):
            self._extended_agent_summary = None

        self._prompt_template = open_person_prompt_template
        self._init_system_message: str | None = None  # initialized later

        ############################################################
        # Special mechanisms used during deserialization
        ############################################################

        # register the agent in the global list of agents
        OpenPerson.add_agent(self)

        self.reset_prompt()

    @staticmethod
    def add_agent(agent: OpenPerson) -> None:
        """
        Adds an agent to the global list of agents. Agent names must be unique,
        so this method will raise an exception if the name is already in use.
        """
        if agent.name in OpenPerson.all_agents:
            raise ValueError(f"Agent name {agent.name} is already in use.")
        OpenPerson.all_agents[agent.name] = agent

    def generate_agent_system_prompt(self) -> str:
        # let's operate on top of a copy of the configuration, because we'll need to add more variables, etc.
        template_variables = self._configuration.copy()

        # Prepare additional action definitions and constraints
        actions_definitions_prompt = ""
        actions_constraints_prompt = ""
        for faculty in self._mental_faculties:
            actions_definitions_prompt += f"{faculty.actions_definitions_prompt()}\n"
            actions_constraints_prompt += f"{faculty.actions_constraints_prompt()}\n"

        # Make the additional prompt pieces available to the template.
        # Identation here is to align with the text structure in the template.
        template_variables["actions_definitions_prompt"] = textwrap.indent(
            actions_definitions_prompt.strip(),
            "  ",
        )
        template_variables["actions_constraints_prompt"] = textwrap.indent(
            actions_constraints_prompt.strip(),
            "  ",
        )

        return chevron.render(self._prompt_template, template_variables)

    def reset_prompt(self) -> None:
        # render the template with the current configuration
        self._init_system_message = self.generate_agent_system_prompt()

        # reset system message
        self.current_messages = [
            {"role": "system", "content": self._init_system_message},
        ]

        # sets up the actual interaction messages to use for prompting
        self.current_messages += self.retrieve_recent_memories()

        # add a final user message, which is neither stimuli or action, to instigate the agent to act properly
        self.current_messages.append(
            {
                "role": "user",
                "content": "Now you **must** generate a sequence of actions following your interaction directives, "
                "and complying with **all** instructions and contraints related to the action you use."
                "DO NOT repeat the exact same action more than once in a row!"
                "These actions **MUST** be rendered following the JSON specification perfectly, including all required keys (even if their value is empty), **ALWAYS**.",
            },
        )

    def define(self, key: str | None, value: Any, group: str | None = None) -> None:
        """
        Define a value to the OpenPerson's configuration.
        If group is None, the value is added to the top level of the configuration.
        Otherwise, the value is added to the specified group.
        """
        # dedent value if it is a string
        if isinstance(value, str):
            value = textwrap.dedent(value)

        if group is None:
            self._configuration[key] = value
        elif key is not None:
            self._configuration[group].append({key: value})
        else:
            self._configuration[group].append(value)

        # must reset prompt after adding to configuration
        self.reset_prompt()

    def define_several(self, group: str | None, records: list) -> None:
        """Define several values to the OpenPerson's configuration, all belonging to the same group."""
        for record in records:
            self.define(key=None, value=record, group=group)

    def retrieve_recent_memories(self, max_content_length: int | None = None) -> list:
        episodes = self.episodic_memory.retrieve_recent()

        if max_content_length is not None:
            episodes = utils.truncate_actions_or_stimuli(episodes, max_content_length)

        return episodes

    def act(  # noqa: C901
        self,
        until_done: bool = True,
        n: int | None = None,
        return_actions: bool = False,
        max_content_length: int = default["max_content_display_length"],
    ) -> list | None:
        """
        Acts in the environment and updates its internal cognitive state.
        Either acts until the agent is done and needs additional stimuli, or acts a fixed number of times,
        but not both.

        Args:
            until_done (bool): Whether to keep acting until the agent is done and needs additional stimuli.
            n (int): The number of actions to perform. Defaults to None.
            return_actions (bool): Whether to return the actions or not. Defaults to False.
            max_content_length: The maximum length of the content. Defaults to default["max_content_display_length"].

        """
        # either act until done or act a fixed number of times, but not both
        if until_done and n is not None:
            raise ValueError(
                "Either act until done or act a fixed number of times, but not both.",
                until_done,
                n,
            )
        if n is not None and n < OpenPerson.MAX_ACTIONS_BEFORE_DONE:
            raise ValueError(
                "n must be less than or equal to the MAX_ACTIONS_BEFORE_DONE.",
                n,
                OpenPerson.MAX_ACTIONS_BEFORE_DONE,
            )

        contents: list = []

        @repeat_on_error(retries=5, exceptions=[KeyError, TypeError])
        def aux_act_once() -> None:
            role, content = self._produce_message()

            cognitive_state = content["cognitive_state"]

            action = content["action"]
            utils.logger.debug(f"{self.name}'s action: {action}")

            self.store_in_memory(
                {
                    "role": role,
                    "content": content,
                    "type": "action",
                    "simulation_timestamp": self.iso_datetime(),
                },
            )

            self._actions_buffer.append(action)
            self._update_cognitive_state(
                goals=cognitive_state["goals"],
                attention=cognitive_state["attention"],
                emotions=cognitive_state["emotions"],
            )

            contents.append(content)
            if OpenPerson.communication_display:
                self._display_communication(
                    role=role,
                    content=content,
                    kind="action",
                    simplified=True,
                    max_content_length=max_content_length,
                )

            #
            # Some actions induce an immediate stimulus or other side-effects. We need to process them here, by means of the mental faculties.
            #
            for faculty in self._mental_faculties:
                faculty.process_action(self, action)

        ##### Option 1: run N actions ######
        if n is not None:
            for _i in range(n):
                aux_act_once()

        ##### Option 2: run until DONE ######
        elif until_done:
            while (len(contents) == 0) or (contents[-1]["action"]["type"] != "DONE"):
                # check if the agent is acting without ever stopping
                if len(contents) > OpenPerson.MAX_ACTIONS_BEFORE_DONE:
                    utils.logger.warning(
                        f"[{self.name}] Agent {self.name} is acting without ever stopping. This may be a bug. Let's stop it here anyway.",
                    )
                    break
                # just some minimum number of actions to check for repetition, could be anything >= 3
                # if the last three actions were the same, then we are probably in a loop
                min_actions_to_check_for_repetition = 4
                if len(contents) > min_actions_to_check_for_repetition and (
                    contents[-1]["action"]
                    == contents[-2]["action"]
                    == contents[-3]["action"]
                ):
                    utils.logger.warning(
                        f"[{self.name}] Agent {self.name} is acting in a loop. This may be a bug. Let's stop it here anyway.",
                    )
                    break

                aux_act_once()

        if return_actions:
            return contents
        return None

    def listen(
        self,
        speech: str,
        source: OpenWorld | OpenPerson | None = None,
        max_content_length: int = default["max_content_display_length"],
    ) -> OpenPerson:
        """
        Listens to another agent (artificial or human) and updates its internal cognitive state.

        Args:
            speech (str): The speech to listen to.
            source (AgentOrWorld, optional): The source of the speech. Defaults to None.
            max_content_length (int, optional): The maximum length of the content. Defaults to default["max_content_display_length"].

        """
        return self._observe(
            stimulus={
                "type": "CONVERSATION",
                "content": speech,
                "source": utils.name_or_empty(source),
            },
            max_content_length=max_content_length,
        )

    def socialize(
        self,
        social_description: str,
        source: OpenWorld | OpenPerson | None = None,
        max_content_length: int = default["max_content_display_length"],
    ) -> OpenPerson:
        """
        Perceives a social stimulus through a description and updates its internal cognitive state.

        Args:
            social_description (str): The description of the social stimulus.
            source (AgentOrWorld, optional): The source of the social stimulus. Defaults to None.
            max_content_length (int, optional): The maximum length of the content. Defaults to default["max_content_display_length"].

        """
        return self._observe(
            stimulus={
                "type": "SOCIAL",
                "content": social_description,
                "source": utils.name_or_empty(source),
            },
            max_content_length=max_content_length,
        )

    def see(
        self,
        visual_description: str,
        source: OpenWorld | OpenPerson | None = None,
        max_content_length: int = default["max_content_display_length"],
    ) -> OpenPerson:
        """
        Perceives a visual stimulus through a description and updates its internal cognitive state.

        Args:
            visual_description (str): The description of the visual stimulus.
            source (AgentOrWorld, optional): The source of the visual stimulus. Defaults to None.
            max_content_length (int, optional): The maximum length of the content. Defaults to default["max_content_display_length"].

        """
        return self._observe(
            stimulus={
                "type": "VISUAL",
                "content": visual_description,
                "source": utils.name_or_empty(source),
            },
            max_content_length=max_content_length,
        )

    def think(
        self,
        thought: str,
        max_content_length: int = default["max_content_display_length"],
    ) -> OpenPerson:
        """Forces the agent to think about something and updates its internal cognitive state."""
        return self._observe(
            stimulus={
                "type": "THOUGHT",
                "content": thought,
                "source": utils.name_or_empty(self),
            },
            max_content_length=max_content_length,
        )

    def _observe(
        self,
        stimulus: dict,
        max_content_length: int = default["max_content_display_length"],
    ) -> OpenPerson:
        stimuli = [stimulus]

        content = {"stimuli": stimuli}

        utils.logger.debug(f"[{self.name}] Observing stimuli: {content}")

        # whatever comes from the outside will be interpreted as coming from 'user', simply because
        # this is the counterpart of 'assistant'

        self.store_in_memory(
            {
                "role": "user",
                "content": content,
                "type": "stimulus",
                "simulation_timestamp": self.iso_datetime(),
            },
        )

        if OpenPerson.communication_display:
            self._display_communication(
                role="user",
                content=content,
                kind="stimuli",
                simplified=True,
                max_content_length=max_content_length,
            )

        return self  # allows easier chaining of methods

    def listen_and_act(
        self,
        speech: str,
        return_actions: bool = False,
        max_content_length: int = 1024,
    ) -> list | None:
        """Convenience method that combines the `listen` and `act` methods."""
        self.listen(speech, max_content_length=max_content_length)
        return self.act(
            return_actions=return_actions,
            max_content_length=max_content_length,
        )

    def see_and_act(
        self,
        visual_description: str,
        return_actions: bool = False,
        max_content_length: int = default["max_content_display_length"],
    ) -> list | None:
        """Convenience method that combines the `see` and `act` methods."""
        self.see(visual_description, max_content_length=max_content_length)
        return self.act(return_actions=return_actions)

    def think_and_act(
        self,
        thought: str,
        return_actions: bool = False,
        max_content_length: int = default["max_content_display_length"],
    ) -> list | None:
        """Convenience method that combines the `think` and `act` methods."""
        self.think(thought, max_content_length=max_content_length)
        return self.act(
            return_actions=return_actions,
            max_content_length=max_content_length,
        )

    def make_agent_accessible(
        self,
        agent: OpenPerson,
        relation_description: str = "An agent I can currently interact with.",
    ) -> None:
        """Makes an agent accessible to this agent."""
        if agent not in self._accessible_agents:
            self._accessible_agents.append(agent)
            self._configuration["currently_accessible_agents"].append(
                {"name": agent.name, "relation_description": relation_description},
            )
        else:
            utils.logger.warning(
                f"[{self.name}] Agent {agent.name} is already accessible to {self.name}.",
            )

    def _produce_message(self) -> tuple[str, dict]:
        # ensure we have the latest prompt (initial system message + selected messages from memory)
        self.reset_prompt()

        messages = [
            {"role": msg["role"], "content": json.dumps(msg["content"])}
            for msg in self.current_messages
        ]

        utils.logger.debug(f"[{self.name}] Sending messages to LLM API")
        utils.logger.debug(f"[{self.name}] Last interaction: {messages[-1]}")

        # TODO: refactor dependency injection
        openai = OpenaiProvider()
        next_message = openai.send_message(
            messages=messages,
            response_format=CognitiveActionModel,
        )

        utils.logger.debug(f"[{self.name}] Received message: {next_message}")
        if next_message is None:
            raise ValueError(
                f"Agent {self.name} received no response from the LLM API.",
            )

        return next_message["role"], utils.extract_json(next_message["content"])

    ###########################################################
    # Internal cognitive state changes
    ###########################################################
    def _update_cognitive_state(
        self,
        goals: str | None = None,
        context: str | None = None,
        attention: str | None = None,
        emotions: str | None = None,
    ) -> None:
        """Update the OpenPerson's cognitive state."""
        # Update current datetime. The passage of time is controlled by the environment, if any.
        if (
            self.environment is not None
            and self.environment.current_datetime is not None
        ):
            self._configuration["current_datetime"] = utils.pretty_datetime(
                self.environment.current_datetime,
            )

        # update current goals
        if goals is not None:
            self._configuration["current_goals"] = goals

        # update current context
        if context is not None:
            self._configuration["current_context"] = context

        # update current attention
        if attention is not None:
            self._configuration["current_attention"] = attention

        # update current emotions
        if emotions is not None:
            self._configuration["current_emotions"] = emotions

        # update relevant memories for the current situation
        current_memory_context = self.retrieve_relevant_memories_for_current_context()
        self._configuration["current_memory_context"] = current_memory_context

        self.reset_prompt()

    ###########################################################
    # Memory management
    ###########################################################
    def store_in_memory(self, value: Any) -> None:
        # TODO find another smarter way to abstract episodic information into semantic memory
        # self.semantic_memory.store(value)

        self.episodic_memory.store(value)

    def retrieve_memories(
        self,
        first_n: int,
        last_n: int,
        include_omission_info: bool = True,
        max_content_length: int | None = None,
    ) -> list:
        episodes = self.episodic_memory.retrieve(
            first_n=first_n,
            last_n=last_n,
            include_omission_info=include_omission_info,
        )

        if max_content_length is not None:
            episodes = utils.truncate_actions_or_stimuli(episodes, max_content_length)

        return episodes

    def retrieve_relevant_memories(
        self,
        relevance_target: str,
        top_k: int = 20,
    ) -> list:
        return self.semantic_memory.retrieve_relevant(relevance_target, top_k=top_k)

    def retrieve_relevant_memories_for_current_context(self, top_k: int = 7) -> list:
        # current context is composed of th recent memories, plus context, goals, attention, and emotions
        context = self._configuration["current_context"]
        goals = self._configuration["current_goals"]
        attention = self._configuration["current_attention"]
        emotions = self._configuration["current_emotions"]
        recent_memories = "\n".join(
            [
                f"  - {m['content']}"
                for m in self.retrieve_memories(
                    first_n=0,
                    last_n=10,
                    max_content_length=100,
                )
            ],
        )

        # put everything together in a nice markdown string to fetch relevant memories
        target = f"""
        Current Context: {context}
        Current Goals: {goals}
        Current Attention: {attention}
        Current Emotions: {emotions}
        Recent Memories:
        {recent_memories}
        """

        utils.logger.debug(
            f"Retrieving relevant memories for contextual target: {target}",
        )

        return self.retrieve_relevant_memories(target, top_k=top_k)

    ###########################################################
    # Inspection conveniences
    ###########################################################
    def _display_communication(
        self,
        role: str,
        content: dict,
        kind: str,
        simplified: bool = True,
        max_content_length: int = default["max_content_display_length"],
    ) -> None:
        """Displays the current communication and stores it in a buffer for later use."""
        if kind == "stimuli":
            rendering = self._pretty_stimuli(
                role=role,
                content=content,
                simplified=simplified,
                max_content_length=max_content_length,
            )
            source = content["stimuli"][0]["source"]
            target = self.name

        elif kind == "action":
            rendering = self._pretty_action(
                role=role,
                content=content,
                simplified=simplified,
                max_content_length=max_content_length,
            )
            source = self.name
            target = content["action"]["target"]

        else:
            msg = f"Unknown communication kind: {kind}"
            raise ValueError(msg)

        # if the agent has no parent environment, then it is a free agent and we can display the communication.
        # otherwise, the environment will display the communication instead. This is important to make sure that
        # the communication is displayed in the correct order, since environments control the flow of their underlying
        # agents.
        if self.environment is None:
            self._push_and_display_latest_communication(
                {
                    "kind": kind,
                    "rendering": rendering,
                    "content": content,
                    "source": source,
                    "target": target,
                },
            )
        else:
            self.environment._push_and_display_latest_communication(  # noqa: SLF001
                {
                    "kind": kind,
                    "rendering": rendering,
                    "content": content,
                    "source": source,
                    "target": target,
                },
            )

    def _push_and_display_latest_communication(self, communication: dict) -> None:
        """Pushes the latest communications to the agent's buffer."""
        self._displayed_communications_buffer.append(communication)
        print(communication["rendering"])

    def pop_latest_actions(self) -> list:
        """
        Returns the latest actions performed by this agent. Typically used
        by an environment to consume the actions and provide the appropriate
        environmental semantics to them (i.e., effects on other agents).
        """
        actions = self._actions_buffer
        self._actions_buffer = []
        return actions

    #############################################################################################
    # Formatting conveniences
    #############################################################################################
    def _pretty_stimuli(
        self,
        role: str,
        content: dict,
        simplified: bool = True,
        max_content_length: int = default["max_content_display_length"],
    ) -> str:
        """Pretty prints stimuli."""
        lines = []
        msg_simplified_actor = "USER"
        for stimus in content["stimuli"]:
            if simplified:
                if stimus["source"] != "":
                    msg_simplified_actor = stimus["source"]

                else:
                    msg_simplified_actor = "USER"

                msg_simplified_type = stimus["type"]
                msg_simplified_content = utils.break_text_at_length(
                    stimus["content"],
                    max_length=max_content_length,
                )

                indent = " " * len(msg_simplified_actor) + "      > "
                msg_simplified_content = textwrap.fill(
                    msg_simplified_content,
                    width=OpenPerson.PP_TEXT_WIDTH,
                    initial_indent=indent,
                    subsequent_indent=indent,
                )

                #
                # Using rich for formatting. Let's make things as readable as possible!
                #

                rich_style = utils.RichTextStyle.get_style_for(
                    "stimulus",
                    msg_simplified_type,
                )
                lines.append(
                    f"[{rich_style}][underline]{msg_simplified_actor}[/] --> [{rich_style}][underline]{self.name}[/]: [{msg_simplified_type}] \n{msg_simplified_content}[/]",
                )
            else:
                lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _pretty_action(
        self,
        role: str,
        content: dict,
        simplified: bool = True,
        max_content_length: int = default["max_content_display_length"],
    ) -> str:
        """Pretty prints an action."""
        if simplified:
            msg_simplified_actor = self.name
            msg_simplified_type = content["action"]["type"]
            msg_simplified_content = utils.break_text_at_length(
                content["action"].get("content", ""),
                max_length=max_content_length,
            )

            indent = " " * len(msg_simplified_actor) + "      > "
            msg_simplified_content = textwrap.fill(
                msg_simplified_content,
                width=OpenPerson.PP_TEXT_WIDTH,
                initial_indent=indent,
                subsequent_indent=indent,
            )

            #
            # Using rich for formatting. Let's make things as readable as possible!
            #
            rich_style = utils.RichTextStyle.get_style_for(
                "action",
                msg_simplified_type,
            )
            return f"[{rich_style}][underline]{msg_simplified_actor}[/] acts: [{msg_simplified_type}] \n{msg_simplified_content}[/]"

        return f"{role}: {content}"

    def iso_datetime(self) -> str | None:
        """
        Returns the current datetime of the environment, if any.

        Returns:
            datetime: The current datetime of the environment in ISO forat.

        """
        if (
            self.environment is not None
            and self.environment.current_datetime is not None
        ):
            return self.environment.current_datetime.isoformat()
        return None

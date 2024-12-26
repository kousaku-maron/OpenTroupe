from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, ClassVar

from rich.console import Console

from opentroupe import utils

if TYPE_CHECKING:
    from opentroupe.agent import OpenPerson


class OpenWorld:
    """Base class for environments."""

    # A dict of all environments created so far.
    all_environments: ClassVar[dict] = {}  # name -> environment

    # Whether to display environments communications or not, for all environments.
    communication_display: bool = True

    def __init__(
        self,
        name: str = "A OpenWorld",
        agents: list[OpenPerson] | None = None,
        initial_datetime: datetime | None = None,
        broadcast_if_no_target: bool = True,
        max_additional_targets_to_display: int = 3,
    ) -> None:
        """
        Initializes an environment.

        Args:
            name (str): The name of the environment.
            agents (list, optional): A list of agents to add to the environment.
            initial_datetime (datetime, optional): The initial datetime of the environment, or None (i.e., explicit time is optional).
                Defaults to the current datetime in the real world.
            broadcast_if_no_target (bool): If True, broadcast actions if the target of an action is not found.
            max_additional_targets_to_display (int): The maximum number of additional targets to display in a communication. If None,
                all additional targets are displayed.

        """
        self.name = name
        self.broadcast_if_no_target = broadcast_if_no_target
        self.simulation_id = None  # will be reset later if the agent is used within a specific simulation scope

        self.agents: list[OpenPerson] = []
        self.name_to_agent: dict[
            str,
            OpenPerson,
        ] = {}  # {agent_name: agent, agent_name_2: agent_2, ...}

        # the buffer of communications that have been displayed so far, used for
        # saving these communications to another output form later (e.g., caching)
        self._displayed_communications_buffer: list = []

        # a temporary buffer for communications target to make rendering easier
        self._target_display_communications_buffer: list = []
        self._max_additional_targets_to_display = max_additional_targets_to_display

        if initial_datetime is not None:
            self.current_datetime = initial_datetime

        if not hasattr(self, "current_datetime"):
            self.current_datetime = datetime.now()  # noqa: DTZ005

        self.console = Console()

        # add the environment to the list of all environments
        OpenWorld.add_environment(self)

        if agents is not None:
            self.add_agents(agents)

    #######################################################################
    # Agent management methods
    #######################################################################
    def add_agents(self, agents: list[OpenPerson]) -> OpenWorld:
        """
        Adds a list of agents to the environment.

        Args:
            agents (list): A list of agents to add to the environment.

        """
        for agent in agents:
            self.add_agent(agent)

        return self  # for chaining

    def add_agent(self, agent: OpenPerson) -> OpenWorld:
        """
        Adds an agent to the environment. The agent must have a unique name within the environment.

        Args:
            agent (OpenPerson): The agent to add to the environment.

        Raises:
            ValueError: If the agent name is not unique within the environment.

        """
        # check if the agent is not already in the environment
        if agent not in self.agents:
            utils.logger.debug(f"Adding agent {agent.name} to the environment.")

            # Agent names must be unique in the environment.
            # Check if the agent name is already there.
            if agent.name not in self.name_to_agent:
                agent.environment = self
                self.agents.append(agent)
                self.name_to_agent[agent.name] = agent
            else:
                raise ValueError(
                    f"Agent names must be unique, but '{agent.name}' is already in the environment.",
                )
        else:
            utils.logger.warning(f"Agent {agent.name} is already in the environment.")

        return self  # for chaining

    def get_agent_by_name(self, name: str) -> OpenPerson | None:
        """
        Returns the agent with the specified name. If no agent with that name exists in the environment,
        returns None.

        Args:
            name (str): The name of the agent to return.

        Returns:
            OpenPerson: The agent with the specified name.

        """
        if name in self.name_to_agent:
            return self.name_to_agent[name]
        return None

    #######################################################################
    # Action handlers
    #
    # Specific actions issued by agents are handled by the environment,
    # because they have effects beyond the agent itself.
    #######################################################################
    def _handle_actions(self, source: OpenPerson, actions: list) -> None:
        """

        Handles the actions issued by the agents.

        Args:
            source (OpenPerson): The agent that issued the actions.
            actions (list): A list of actions issued by the agents. Each action is actually a
              JSON specification.

        """
        for action in actions:
            action_type = action["type"]  # this is the only required field
            content = action.get("content", None)
            target = action.get("target", None)

            utils.logger.debug(
                f"[{self.name}] Handling action {action_type} from agent {utils.name_or_empty(source)}. Content: {content}, target: {target}.",
            )

            # only some actions require the enviroment to intervene
            if action_type == "REACH_OUT":
                self._handle_reach_out(source, content, target)
            elif action_type == "TALK":
                self._handle_talk(source, content, target)

    def _handle_reach_out(
        self,
        source_agent: OpenPerson,
        content: str,  # noqa: ARG002
        target: str,
    ) -> None:
        """
        Handles the REACH_OUT action. This default implementation always allows REACH_OUT to succeed.
        Subclasses might override this method to implement different policies.

        Args:
            source_agent (OpenPerson): The agent that issued the REACH_OUT action.
            content (str): The content of the message.
            target (str): The target of the message.

        """
        # This default implementation always allows REACH_OUT to suceed.
        target_agent = self.get_agent_by_name(target)

        if target_agent is not None:
            source_agent.make_agent_accessible(target_agent)
            target_agent.make_agent_accessible(source_agent)

            source_agent.socialize(
                f"{utils.name_or_empty(target_agent)} was successfully reached out, and is now available for interaction.",
                source=self,
            )
            target_agent.socialize(
                f"{utils.name_or_empty(source_agent)} reached out to you, and is now available for interaction.",
                source=self,
            )

        else:
            utils.logger.debug(
                f"[{self.name}] REACH_OUT action failed: target agent '{target}' not found.",
            )

    def _handle_talk(self, source_agent: OpenPerson, content: str, target: str) -> None:
        """
        Handles the TALK action by delivering the specified content to the specified target.

        Args:
            source_agent (OpenPerson): The agent that issued the TALK action.
            content (str): The content of the message.
            target (str, optional): The target of the message.

        """
        target_agent = self.get_agent_by_name(target)

        utils.logger.debug(
            f"[{self.name}] Delivering message from {utils.name_or_empty(source_agent)} to {utils.name_or_empty(target_agent)}.",
        )

        if target_agent is not None:
            target_agent.listen(content, source=source_agent)
        elif self.broadcast_if_no_target:
            self.broadcast(content, source=source_agent)

    #######################################################################
    # Simulation control methods
    #######################################################################
    def _step(self, timedelta_per_step: timedelta | None = None) -> dict:
        """
        Performs a single step in the environment. This default implementation
        simply calls makes all agents in the environment act and properly
        handle the resulting actions. Subclasses might override this method to implement
        different policies.
        """
        # increase current datetime if timedelta is given. This must happen before
        # any other simulation updates, to make sure that the agents are acting
        # in the correct time, particularly if only one step is being run.
        self._advance_datetime(timedelta_per_step)

        # agents can act
        agents_actions: dict = {}
        for agent in self.agents:
            utils.logger.debug(
                f"[{self.name}] Agent {utils.name_or_empty(agent)} is acting.",
            )
            actions = agent.act(return_actions=True)
            agents_actions[agent.name] = actions

            self._handle_actions(agent, agent.pop_latest_actions())

        return agents_actions

    def _advance_datetime(self, timedelta: timedelta | None) -> None:
        """
        Advances the current datetime of the environment by the specified timedelta.

        Args:
            timedelta (timedelta): The timedelta to advance the current datetime by.

        """
        if timedelta is not None:
            self.current_datetime += timedelta
        else:
            utils.logger.info(
                f"[{self.name}] No timedelta provided, so the datetime was not advanced.",
            )

    def run(
        self,
        steps: int,
        timedelta_per_step: timedelta | None = None,
        return_actions: bool = False,
    ) -> None | list:
        """
        Runs the environment for a given number of steps.

        Args:
            steps (int): The number of steps to run the environment for.
            timedelta_per_step (timedelta, optional): The time interval between steps. Defaults to None.
            return_actions (bool, optional): If True, returns the actions taken by the agents. Defaults to False.

        Returns:
            list: A list of actions taken by the agents over time, if return_actions is True. The list has this format:
                  [{agent_name: [action_1, action_2, ...]}, {agent_name_2: [action_1, action_2, ...]}, ...]

        """
        agents_actions_over_time = []
        for i in range(steps):
            utils.logger.info(
                f"[{self.name}] Running world  simulation step {i+1} of {steps}.",
            )

            if OpenWorld.communication_display:
                self._display_communication(
                    cur_step=i + 1,
                    total_steps=steps,
                    kind="step",
                    timedelta_per_step=timedelta_per_step,
                )

            agents_actions = self._step(timedelta_per_step=timedelta_per_step)
            agents_actions_over_time.append(agents_actions)

        if return_actions:
            return agents_actions_over_time
        return None

    #######################################################################
    # Interaction methods
    #######################################################################
    def broadcast(
        self,
        speech: str,
        source: OpenWorld | OpenPerson | None = None,
    ) -> None:
        """
        Delivers a speech to all agents in the environment.

        Args:
            speech (str): The content of the message.
            source (AgentOrWorld, optional): The agent or environment that issued the message. Defaults to None.

        """
        utils.logger.debug(f"[{self.name}] Broadcasting message: '{speech}'.")

        for agent in self.agents:
            # do not deliver the message to the source
            if agent != source:
                agent.listen(speech, source=source)

    ###########################################################
    # Formatting conveniences
    ###########################################################
    def _display_communication(
        self,
        cur_step: int,
        total_steps: int,
        kind: str,
        timedelta_per_step: timedelta | None = None,
    ) -> None:
        """Displays the current communication and stores it in a buffer for later use."""
        if kind == "step":
            rendering = self._pretty_step(
                cur_step=cur_step,
                total_steps=total_steps,
                timedelta_per_step=timedelta_per_step,
            )
        else:
            raise ValueError(f"Unknown communication kind: {kind}")

        self._push_and_display_latest_communication(
            {
                "kind": kind,
                "rendering": rendering,
                "content": None,
                "source": None,
                "target": None,
            },
        )

    def _push_and_display_latest_communication(self, communication: dict) -> None:  # noqa: C901, PLR0912
        """Pushes the latest communications to the agent's buffer."""
        #
        # check if the communication is just repeating the last one for a different target
        #
        if len(self._displayed_communications_buffer) > 0:
            # get values from last communication
            last_communication = self._displayed_communications_buffer[-1]
            last_kind = last_communication["kind"]
            last_source = last_communication["source"]
            if last_kind == "action":
                last_content = last_communication["content"]["action"]["content"]
                last_type = last_communication["content"]["action"]["type"]
            elif last_kind == "stimulus":
                last_content = last_communication["content"]["stimulus"]["content"]
                last_type = last_communication["content"]["stimulus"]["type"]
            elif last_kind == "stimuli":
                last_stimulus = last_communication["content"]["stimuli"][0]
                last_content = last_stimulus["content"]
                last_type = last_stimulus["type"]
            else:
                last_content = None
                last_type = None

            # get values from current communication
            current_kind = communication["kind"]
            current_target = communication["target"]
            current_source = communication["source"]
            if current_kind == "action":
                current_content = communication["content"]["action"]["content"]
                current_type = communication["content"]["action"]["type"]
            elif current_kind == "stimulus":
                current_content = communication["content"]["stimulus"]["content"]
                current_type = communication["content"]["stimulus"]["type"]
            elif current_kind == "stimuli":
                current_stimulus = communication["content"]["stimuli"][0]
                current_content = current_stimulus["content"]
                current_type = current_stimulus["type"]
            else:
                current_content = None
                current_type = None

            # if we are repeating the last communication, let's simplify the rendering
            if (
                (last_source == current_source)
                and (last_type == current_type)
                and (last_kind == current_kind)
                and (last_content is not None)
                and (last_content == current_content)
                and (current_target is not None)
            ):
                self._target_display_communications_buffer.append(current_target)

                rich_style = utils.RichTextStyle.get_style_for(last_kind, last_type)

                # print the additional target a limited number of times if a max is set, or
                # always if no max is set.
                if (self._max_additional_targets_to_display is None) or len(
                    self._target_display_communications_buffer,
                ) < self._max_additional_targets_to_display:
                    communication["rendering"] = (
                        " " * len(last_source)
                        + f"[{rich_style}]       + --> [underline]{current_target}[/][/]"
                    )

                elif (
                    len(self._target_display_communications_buffer)
                    == self._max_additional_targets_to_display
                ):
                    communication["rendering"] = (
                        " " * len(last_source)
                        + f"[{rich_style}]       + --> ...others...[/]"
                    )

                else:  # don't display anything anymore
                    communication["rendering"] = None

            else:
                # no repetition, so just display the communication and reset the targets buffer
                self._target_display_communications_buffer = []  # resets

        else:
            # no repetition, so just display the communication and reset the targets buffer
            self._target_display_communications_buffer = []  # resets

        self._displayed_communications_buffer.append(communication)
        self._display(communication)

    def _display(self, communication: dict) -> None:
        # unpack the rendering to find more info
        content = communication["rendering"]
        kind = communication["kind"]

        if content is not None:
            # render as appropriate
            if kind == "step":
                self.console.rule(content)
            else:
                self.console.print(content)

    def _pretty_step(
        self,
        cur_step: int,
        total_steps: int,
        timedelta_per_step: timedelta | None = None,
    ) -> str:
        rendering = f"{self.name} step {cur_step} of {total_steps}"
        if timedelta_per_step is not None:
            rendering += f" ({utils.pretty_datetime(self.current_datetime)})"

        return rendering

    #######################################################################
    # IO
    #######################################################################

    @staticmethod
    def add_environment(environment: OpenWorld) -> None:
        """
        Adds an environment to the list of all environments. Environment names must be unique,
        so if an environment with the same name already exists, an error is raised.
        """
        if environment.name in OpenWorld.all_environments:
            raise ValueError(
                f"Environment names must be unique, but '{environment.name}' is already defined.",
            )
        OpenWorld.all_environments[environment.name] = environment

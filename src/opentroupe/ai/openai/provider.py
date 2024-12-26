import os
import time
from typing import Any, TypedDict, cast

from openai import (
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletion

from opentroupe import utils
from opentroupe.ai.provider import (
    NextMessage,
    Provider,
)
from opentroupe.ai.provider import (
    default as provider_default,
)

from .exception import InvalidRequestError, NonTerminalError

###########################################################################
# Default parameter values
###########################################################################


class DefaultParameters(TypedDict):
    model: str
    temperature: float
    max_tokens: int
    top_p: int
    frequency_penalty: float
    presence_penalty: float
    timeout: float


default: DefaultParameters = {
    "model": "gpt-4o-mini",
    "temperature": 1.5,
    "max_tokens": 1024,
    "top_p": 0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "timeout": 60.0,
}


class OpenaiProvider(Provider):
    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize the OpenAI provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenAI API key is missing. Please provide it in the config or set the OPENAI_API_KEY environment variable.",
            )

        # Pass the entire config to the OpenAI client constructor
        self.client = OpenAI(api_key=api_key)

    def _raw_model_call(self, chat_api_params: dict) -> ChatCompletion:
        """
        Calls the OpenAI API with the given parameters. Subclasses should
        override this method to implement their own API calls.
        """
        if "response_format" in chat_api_params:
            # to enforce the response format, we need to use a different method

            del chat_api_params["stream"]

            return self.client.beta.chat.completions.parse(
                **chat_api_params,
            )

        return self.client.chat.completions.create(
            **chat_api_params,
        )

    def send_message(  # noqa: C901
        self,
        messages: list,
        max_attempts: int = provider_default["max_attempts"],
        waiting_time: int = provider_default["waiting_time"],
        exponential_backoff_factor: int = provider_default[
            "exponential_backoff_factor"
        ],
        response_format: Any | None = None,
    ) -> NextMessage | None:
        def aux_exponential_backoff() -> None:
            nonlocal waiting_time

            # in case waiting time was initially set to 0
            if waiting_time <= 0:
                waiting_time = 2

            utils.logger.info(
                f"Request failed. Waiting {waiting_time} seconds between requests...",
            )
            time.sleep(waiting_time)

            # exponential backoff
            waiting_time = waiting_time * exponential_backoff_factor

        # We need to adapt the parameters to the API type, so we create a dictionary with them first
        chat_api_params = {
            "model": default["model"],
            "messages": messages,
            "temperature": default["temperature"],
            "max_tokens": default["max_tokens"],
            "top_p": default["top_p"],
            "frequency_penalty": default["frequency_penalty"],
            "presence_penalty": default["presence_penalty"],
            "stop": [],
            "timeout": default["timeout"],
            "stream": False,
            "n": 1,
        }

        if response_format is not None:
            chat_api_params["response_format"] = response_format

        i = 0
        while i < max_attempts:
            try:
                i += 1

                start_time = time.monotonic()
                utils.logger.debug(
                    f"Calling model with client class {self.__class__.__name__}.",
                )

                ###############################################################
                # call the model, either from the cache or from the API
                ###############################################################

                if waiting_time > 0:
                    utils.logger.info(
                        f"Waiting {waiting_time} seconds before next API request (to avoid throttling)...",
                    )
                    time.sleep(waiting_time)
                    response = self._raw_model_call(chat_api_params)

                    utils.logger.debug(f"Got response from API: {response}")
                    end_time = time.monotonic()
                    utils.logger.debug(
                        f"Got response in {end_time - start_time:.2f} seconds after {i} attempts.",
                    )

                    response_dict = response.choices[0].message.to_dict()
                    return cast(NextMessage, utils.sanitize_dict(response_dict))

            except InvalidRequestError as e:  # noqa: PERF203
                utils.logger.error(f"[{i}] Invalid request error, won't retry: {e}")

                # there's no point in retrying if the request is invalid
                # so we return None right away
                return None

            except BadRequestError as e:
                utils.logger.error(f"[{i}] Invalid request error, won't retry: {e}")

                # there's no point in retrying if the request is invalid
                # so we return None right away
                return None

            except RateLimitError:
                utils.logger.warning(
                    f"[{i}] Rate limit error, waiting a bit and trying again.",
                )
                aux_exponential_backoff()

            except NonTerminalError as e:
                utils.logger.error(f"[{i}] Non-terminal error: {e}")
                aux_exponential_backoff()

            except Exception as e:  # noqa: BLE001
                utils.logger.error(f"[{i}] Error: {e}")
        return None

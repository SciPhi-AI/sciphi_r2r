import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

from .base import Provider, ProviderConfig

logger = logging.getLogger(__name__)


class PromptConfig(ProviderConfig):
    default_system_name: Optional[str] = "default_system"
    default_task_name: Optional[str] = "default_rag"

    # TODO - Replace this with a database
    file_path: Path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "providers",
        "prompts",
        "defaults.jsonl",
    )

    def validate(self) -> None:
        pass

    @property
    def supported_providers(self) -> list[str]:
        # Return a list of supported prompt providers
        return ["r2r"]


class PromptProvider(Provider):
    def __init__(self, config: Optional[PromptConfig] = None):
        if config is None:
            config = PromptConfig()
        elif not isinstance(config, PromptConfig):
            raise ValueError(
                "PromptProvider must be initialized with a `PromptConfig`."
            )
        logger.info(f"Initializing PromptProvider with config {config}.")
        super().__init__(config)

    @abstractmethod
    def add_prompt(
        self, name: str, template: str, input_types: dict[str, str]
    ) -> None:
        pass

    @abstractmethod
    def get_prompt(
        self, prompt_name: str, inputs: Optional[dict[str, Any]] = None
    ) -> str:
        pass

    @abstractmethod
    def get_all_prompts(self) -> dict[str, str]:
        pass

    @abstractmethod
    def update_prompt(
        self,
        name: str,
        template: Optional[str] = None,
        input_types: Optional[dict[str, str]] = None,
    ) -> None:
        pass

    def _get_message_payload(
        self,
        system_prompt_name: Optional[str] = None,
        system_role: str = "system",
        system_inputs: dict = {},
        task_prompt_name: Optional[str] = None,
        task_role: str = "user",
        task_inputs: dict = {},
    ) -> dict:
        system_prompt = self.get_prompt(
            system_prompt_name or self.config.default_system_name,
            system_inputs,
        )
        task_prompt = self.get_prompt(
            task_prompt_name or self.config.default_task_name, task_inputs
        )
        return [
            {
                "role": system_role,
                "content": system_prompt,
            },
            {"role": task_role, "content": task_prompt},
        ]

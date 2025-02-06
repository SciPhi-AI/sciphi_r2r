import logging

from core.base.providers.embedding import EmbeddingConfig, EmbeddingProvider

from .litellm import LiteLLMEmbeddingProvider
from .ollama import OllamaEmbeddingProvider
from .openai import OpenAIEmbeddingProvider

logger = logging.getLogger(__name__)


class R2REmbeddingProvider(EmbeddingProvider):
    """
    A provider that routes to the right embedding provider:
    - If model starts with "openai/", use OpenAIEmbeddingProvider
    - If model starts with "ollama/", use OllamaEmbeddingProvider
    - Otherwise, fallback to LiteLLMEmbeddingProvider
    """

    def __init__(self, config: EmbeddingConfig, *args, **kwargs) -> None:
        super().__init__(config)
        self.config = config

        logger.info("Initializing R2REmbeddingProvider...")
        self._litellm_provider = LiteLLMEmbeddingProvider(
            self.config, *args, **kwargs
        )
        self._ollama_provider = OllamaEmbeddingProvider(
            self.config, *args, **kwargs
        )
        self._openai_provider = OpenAIEmbeddingProvider(
            self.config, *args, **kwargs
        )

        logger.debug(
            "R2REmbeddingProvider initialized with OpenAI, Ollama, and LiteLLM sub-providers."
        )

    def _choose_subprovider_by_model(
        self, model_name: str
    ) -> EmbeddingProvider:
        """
        Decide which underlying sub-provider to call based on the model name.
        """
        if model_name.startswith("azure/") or model_name.startswith(
            "lm_studio/"
        ):
            return self._litellm_provider
        elif model_name.startswith("openai/"):
            return self._openai_provider
        elif model_name.startswith("ollama/"):
            return self._ollama_provider
        return self._litellm_provider

    async def get_embeddings(
        self, texts: list[str], model_type: str = "inference"
    ) -> list[list[float]]:
        """
        Get embeddings using the appropriate sub-provider based on the model name.
        """
        model_name = (
            self.config.inference_model
            if model_type == "inference"
            else self.config.ingestion_model
        )
        sub_provider = self._choose_subprovider_by_model(model_name)
        return await sub_provider.get_embeddings(texts, model_type)

import logging
import os

from openai import OpenAI

from r2r.core import EmbeddingProvider, PipelineStage, VectorSearchResult

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    MODEL_TO_TOKENIZER = {
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
    }
    MODEL_TO_DIMENSIONS = {
        "text-embedding-ada-002": [1536],
        "text-embedding-3-small": [512, 1536],
        "text-embedding-3-large": [256, 1024, 3072],
    }

    def __init__(self, config: dict):
        logger.info(
            "Initializing `OpenAIEmbeddingProvider` to provide embeddings."
        )
        super().__init__(config)
        provider = config.get("provider", None)
        if not provider:
            raise ValueError(
                "Must set provider in order to initialize SentenceTransformerEmbeddingProvider."
            )

        if provider != "openai":
            raise ValueError(
                "OpenAIEmbeddingProvider must be initialized with provider `openai`."
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "Must set OPENAI_API_KEY in order to initialize OpenAIEmbeddingProvider."
            )
        self.client = OpenAI()
        if config.get("rerank_model", None):
            raise ValueError(
                "OpenAIEmbeddingProvider does not support separate reranking."
            )
        self.search_model = config.get("search_model", None)
        self.search_dimension = config.get("search_dimension", None)

        if self.search_model not in OpenAIEmbeddingProvider.MODEL_TO_TOKENIZER:
            raise ValueError(
                f"OpenAI embedding model {self.search_model} not supported."
            )
        if (
            self.search_dimension
            and self.search_dimension
            not in OpenAIEmbeddingProvider.MODEL_TO_DIMENSIONS[
                self.search_model
            ]
        ):
            raise ValueError(
                f"Dimensions {self.dimension} for {self.search_model} are not supported"
            )

        if not self.search_model or not self.search_dimension:
            raise ValueError(
                "Must set search_model and search_dimension in order to initialize OpenAIEmbeddingProvider."
            )

        self.rerank_model = config.get("rerank_model", None)
        if self.rerank_model:
            raise ValueError(
                "OpenAIEmbeddingProvider does not support separate reranking."
            )

    def get_embedding(self, text: str, stage: PipelineStage = PipelineStage.SEARCH) -> list[float]:
        if stage != PipelineStage.SEARCH:
            raise ValueError(
                "OpenAIEmbeddingProvider only supports search stage."
            )

        return (
            self.client.embeddings.create(
                input=[text],
                model=self.search_model,
                dimensions=self.search_dimension
                or OpenAIEmbeddingProvider.MODEL_TO_DIMENSIONS[
                    self.search_model
                ][-1],
            )
            .data[0]
            .embedding
        )

    def get_embeddings(
        self, texts: list[str], stage: PipelineStage = PipelineStage.SEARCH
    ) -> list[list[float]]:
        if stage != PipelineStage.SEARCH:
            raise ValueError(
                "OpenAIEmbeddingProvider only supports search stage."
            )

        return [
            ele.embedding
            for ele in self.client.embeddings.create(
                input=texts,
                model=self.search_model,
                dimensions=self.search_dimension
                or OpenAIEmbeddingProvider.MODEL_TO_DIMENSIONS[
                    self.search_model
                ][-1],
            ).data
        ]

    def rerank(
        self,
        transformed_query: str,
        texts: list[VectorSearchResult],
        stage: PipelineStage = PipelineStage.RERANK,
        limit: int = 10,
    ):
        return texts[:limit]

    def tokenize_string(self, text: str, model: str) -> list[int]:
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Must download tiktoken library to run `tokenize_string`."
            )
        # tiktoken encoding -
        # cl100k_base -	gpt-4, gpt-3.5-turbo, text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
        if model not in OpenAIEmbeddingProvider.MODEL_TO_TOKENIZER:
            raise ValueError(f"OpenAI embedding model {model} not supported.")
        encoding = tiktoken.get_encoding(
            OpenAIEmbeddingProvider.MODEL_TO_TOKENIZER[model]
        )
        return encoding.encode(text)

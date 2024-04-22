import logging
from typing import Any

import dotenv

from r2r.core import (
    EmbeddingConfig,
    EvalConfig,
    LLMConfig,
    LoggingDatabaseConnection,
)
from r2r.core.utils import RecursiveCharacterTextSplitter
from r2r.llms import (
    LiteLLM,
    LiteLLMConfig,
    LlamaCPP,
    LlamaCppConfig,
    OpenAILLM,
)
from r2r.pipelines import (
    BasicEmbeddingPipeline,
    BasicEvalPipeline,
    BasicIngestionPipeline,
    BasicScraperPipeline,
    QnARAGPipeline,
)

from .app import create_app
from .utils import R2RConfig

dotenv.load_dotenv()


class E2EPipelineFactory:
    @staticmethod
    def get_vector_db_provider(database_config: dict[str, Any]):
        if database_config["provider"] == "qdrant":
            from r2r.vector_dbs import QdrantDB

            return QdrantDB()
        elif database_config["provider"] == "pgvector":
            from r2r.vector_dbs import PGVectorDB

            return PGVectorDB()
        elif database_config["provider"] == "local":
            from r2r.vector_dbs import LocalVectorDB

            return LocalVectorDB()

    @staticmethod
    def get_embedding_provider(embedding_config: dict[str, Any]):
        if embedding_config["provider"] == "openai":
            from r2r.embeddings import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider(
                EmbeddingConfig.create(**embedding_config)
            )
        elif embedding_config["provider"] == "sentence-transformers":
            from r2r.embeddings import SentenceTransformerEmbeddingProvider

            return SentenceTransformerEmbeddingProvider(
                EmbeddingConfig.create(**embedding_config)
            )
        else:
            raise ValueError(
                f"Embedding provider {embedding_config['provider']} not supported"
            )

    @staticmethod
    def get_eval_provider(eval_config: dict[str, Any]):
        eval_config = EvalConfig.create(**eval_config)
        if eval_config.provider == "deepeval":
            try:
                from r2r.eval import DeepEvalProvider
            except ImportError:
                raise ImportError(
                    "DeepEval is not installed. Please install it using `pip install deepeval`."
                )
            eval_provider = DeepEvalProvider(eval_config)

        elif eval_config.provider == "parea":
            try:
                from r2r.eval import PareaEvalProvider
            except ImportError:
                raise ImportError(
                    "Parea is not installed. Please install it using `pip install parea-ai`."
                )
            eval_provider = PareaEvalProvider(eval_config)
        elif eval_config.provider == "none":
            eval_provider = None
        return eval_provider

    @staticmethod
    def get_llm_provider(llm_config: dict[str, Any]):
        if llm_config["provider"] == "openai":
            return OpenAILLM(LLMConfig(provider="openai"))
        elif llm_config["provider"] == "litellm":
            return LiteLLM(LiteLLMConfig())
        elif llm_config["provider"] == "llama-cpp":
            return LlamaCPP(
                LlamaCppConfig(
                    llm_config.get("model_path", ""),
                    llm_config.get("model_name", ""),
                )
            )

    @staticmethod
    def get_text_splitter(text_splitter_config: dict[str, Any]):
        if text_splitter_config["type"] != "recursive_character":
            raise ValueError(
                "Only recursive character text splitter is supported"
            )
        return RecursiveCharacterTextSplitter(
            chunk_size=text_splitter_config["chunk_size"],
            chunk_overlap=text_splitter_config["chunk_overlap"],
            length_function=len,
            is_separator_regex=False,
        )

    @staticmethod
    def create_pipeline(
        config: R2RConfig,
        vector_db_provider=None,
        embedding_provider=None,
        llm_provider=None,
        text_splitter=None,
        adapters=None,
        scraper_pipeline_impl=BasicScraperPipeline,
        ingestion_pipeline_impl=BasicIngestionPipeline,
        embedding_pipeline_impl=BasicEmbeddingPipeline,
        rag_pipeline_impl=QnARAGPipeline,
        eval_pipeline_impl=BasicEvalPipeline,
        app_fn=create_app,
    ):
        logging.basicConfig(level=config.logging_database["level"])

        embedding_provider = (
            embedding_provider
            or E2EPipelineFactory.get_embedding_provider(config.embedding)
        )
        llm_provider = llm_provider or E2EPipelineFactory.get_llm_provider(
            config.language_model
        )
        vector_db_provider = (
            vector_db_provider
            or E2EPipelineFactory.get_vector_db_provider(
                config.vector_database
            )
        )
        collection_name = config.vector_database["collection_name"]
        vector_db_provider.initialize_collection(
            collection_name, embedding_provider.search_dimension
        )

        eval_provider = E2EPipelineFactory.get_eval_provider(config.evals)

        logging_connection = LoggingDatabaseConnection(
            config.logging_database["provider"],
            config.logging_database["collection_name"],
        )

        text_splitter = text_splitter or E2EPipelineFactory.get_text_splitter(
            config.ingestion["text_splitter"]
        )

        scrpr_pipeline = scraper_pipeline_impl()
        ingst_pipeline = ingestion_pipeline_impl(adapters=adapters)
        embd_pipeline = embedding_pipeline_impl(
            embedding_provider=embedding_provider,
            vector_db_provider=vector_db_provider,
            logging_connection=logging_connection,
            text_splitter=text_splitter,
            embedding_batch_size=config.embedding.get("batch_size", 1),
        )
        cmpl_pipeline = rag_pipeline_impl(
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            vector_db_provider=vector_db_provider,
            logging_connection=logging_connection,
        )
        eval_pipeline = eval_pipeline_impl(
            eval_provider, logging_connection=logging_connection
        )

        app = app_fn(
            scraper_pipeline=scrpr_pipeline,
            ingestion_pipeline=ingst_pipeline,
            embedding_pipeline=embd_pipeline,
            rag_pipeline=cmpl_pipeline,
            eval_pipeline=eval_pipeline,
            config=config,
            logging_connection=logging_connection,
        )

        return app

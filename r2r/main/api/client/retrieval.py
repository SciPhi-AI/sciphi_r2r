from typing import AsyncGenerator, Optional, Union

from r2r.base import (
    GenerationConfig,
    KGSearchSettings,
    Message,
    VectorSearchSettings,
)
from r2r.base.api.models import RAGResponse, SearchResponse


class RetrievalMethods:
    @staticmethod
    async def search(
        client,
        query: str,
        vector_search_settings: Optional[
            Union[dict, VectorSearchSettings]
        ] = None,
        kg_search_settings: Optional[Union[dict, KGSearchSettings]] = None,
        *args,
        **kwargs
    ) -> SearchResponse:
        if isinstance(vector_search_settings, dict):
            vector_search_settings = VectorSearchSettings(
                **vector_search_settings
            )
        if isinstance(kg_search_settings, dict):
            kg_search_settings = KGSearchSettings(**kg_search_settings)

        data = {
            "query": query,
            "vector_search_settings": (
                vector_search_settings.dict()
                if vector_search_settings
                else None
            ),
            "kg_search_settings": (
                kg_search_settings.dict() if kg_search_settings else None
            ),
        }
        response = await client._make_request("POST", "search", json=data)
        return response

    @staticmethod
    async def rag(
        client,
        query: str,
        rag_generation_config: Union[dict, GenerationConfig],
        vector_search_settings: Optional[
            Union[dict, VectorSearchSettings]
        ] = None,
        kg_search_settings: Optional[Union[dict, KGSearchSettings]] = None,
        *args,
        **kwargs
    ) -> Union[RAGResponse, AsyncGenerator[RAGResponse, None]]:
        if isinstance(rag_generation_config, dict):
            rag_generation_config = GenerationConfig(**rag_generation_config)
        if isinstance(vector_search_settings, dict):
            vector_search_settings = VectorSearchSettings(
                **vector_search_settings
            )
        if isinstance(kg_search_settings, dict):
            kg_search_settings = KGSearchSettings(**kg_search_settings)

        data = {
            "query": query,
            "rag_generation_config": rag_generation_config.dict(),
            "vector_search_settings": (
                vector_search_settings.dict()
                if vector_search_settings
                else None
            ),
            "kg_search_settings": (
                kg_search_settings.dict() if kg_search_settings else None
            ),
        }

        if rag_generation_config.stream:

            async def stream_response():
                async for chunk in await client._make_request(
                    "POST", "rag", json=data, stream=True
                ):
                    yield RAGResponse(**chunk)

            return stream_response()
        else:
            response = await client._make_request("POST", "rag", json=data)
            return response

    @staticmethod
    async def agent(
        client,
        messages: list[Union[dict, Message]],
        rag_generation_config: Union[dict, GenerationConfig],
        vector_search_settings: Optional[
            Union[dict, VectorSearchSettings]
        ] = None,
        kg_search_settings: Optional[Union[dict, KGSearchSettings]] = None,
        task_prompt_override: Optional[str] = None,
        include_title_if_available: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Union[list[Message], AsyncGenerator[Message, None]]:
        if isinstance(rag_generation_config, dict):
            rag_generation_config = GenerationConfig(**rag_generation_config)
        if isinstance(vector_search_settings, dict):
            vector_search_settings = VectorSearchSettings(
                **vector_search_settings
            )
        if isinstance(kg_search_settings, dict):
            kg_search_settings = KGSearchSettings(**kg_search_settings)

        messages = [
            Message(**msg) if isinstance(msg, dict) else msg
            for msg in messages
        ]

        data = {
            "messages": [msg.model_dump() for msg in messages],
            "rag_generation_config": rag_generation_config.dict(),
            "vector_search_settings": (
                vector_search_settings.dict()
                if vector_search_settings
                else None
            ),
            "kg_search_settings": (
                kg_search_settings.dict() if kg_search_settings else None
            ),
            "task_prompt_override": task_prompt_override,
            "include_title_if_available": include_title_if_available,
        }

        if rag_generation_config.stream:

            async def stream_response():
                async for chunk in await client._make_request(
                    "POST", "agent", json=data, stream=True
                ):
                    yield Message(**chunk)

            return stream_response()
        else:
            response = await client._make_request("POST", "agent", json=data)
            return response

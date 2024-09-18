import json
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Generator, Optional
from uuid import UUID

from core.base import (
    AsyncState,
    CompletionProvider,
    CompletionRecord,
    LLMChatCompletionChunk,
    PipeType,
    PromptProvider,
    format_search_results_for_stream
)
from core.base.abstractions.llm import GenerationConfig

from ..abstractions.generator_pipe import GeneratorPipe
from .search_rag_pipe import SearchRAGPipe

logger = logging.getLogger(__name__)


class StreamingSearchRAGPipe(SearchRAGPipe):
    VECTOR_SEARCH_STREAM_MARKER = (
        "search"  # TODO - change this to vector_search in next major release
    )
    KG_LOCAL_SEARCH_STREAM_MARKER = "kg_local_search"
    KG_GLOBAL_SEARCH_STREAM_MARKER = "kg_global_search"
    COMPLETION_STREAM_MARKER = "completion"

    def __init__(
        self,
        llm_provider: CompletionProvider,
        prompt_provider: PromptProvider,
        type: PipeType = PipeType.GENERATOR,
        config: Optional[GeneratorPipe] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            llm_provider=llm_provider,
            prompt_provider=prompt_provider,
            type=type,
            config=config
            or GeneratorPipe.Config(
                name="default_streaming_rag_pipe", task_prompt="default_rag"
            ),
            *args,
            **kwargs,
        )

    async def _run_logic(
        self,
        input: SearchRAGPipe.Input,
        state: AsyncState,
        rag_generation_config: GenerationConfig,
        completion_record: Optional[CompletionRecord] = None,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        run_id = kwargs.get("run_id")
        context = ""
        async for query, search_results in input.message:
            result = format_search_results_for_stream(search_results)
            yield result
            context += result

        messages = self.prompt_provider._get_message_payload(
            system_prompt_name=self.config.system_prompt,
            task_prompt_name=self.config.task_prompt,
            task_inputs={"query": query, "context": context},
        )
        yield f"<{self.COMPLETION_STREAM_MARKER}>"

        response = ""
        for chunk in self.llm_provider.get_completion_stream(
            messages=messages, generation_config=rag_generation_config
        ):
            chunk = StreamingSearchRAGPipe._process_chunk(chunk)
            response += chunk
            yield chunk

        yield f"</{self.COMPLETION_STREAM_MARKER}>"

        completion_record.search_results = search_results
        completion_record.llm_response = response
        completion_record.completion_end_time = datetime.now()
        await self.log_completion_record(run_id, completion_record)

    async def _yield_chunks(
        self,
        start_marker: str,
        chunks: Generator[str, None, None],
        end_marker: str,
    ) -> str:
        yield start_marker
        for chunk in chunks:
            yield chunk
        yield end_marker

    @staticmethod
    def _process_chunk(chunk: LLMChatCompletionChunk) -> str:
        return chunk.choices[0].delta.content or ""

    async def log_completion_record(
        self, run_id: UUID, completion_record: CompletionRecord
    ):
        await self.enqueue_log(
            run_id=run_id,
            key="completion_record",
            value=completion_record.to_json(),
        )

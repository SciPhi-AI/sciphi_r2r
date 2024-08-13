from typing import Any, AsyncGenerator, Optional, Union

from r2r.base import (
    AsyncState,
    ChunkingProvider,
    DocumentExtraction,
    DocumentFragment,
    PipeType,
    R2RDocumentProcessingError,
    RunLoggingSingleton,
    generate_id_from_label,
)
from r2r.base.pipes.base_pipe import AsyncPipe


class ChunkingPipe(AsyncPipe):
    class Input(AsyncPipe.Input):
        message: AsyncGenerator[
            Union[DocumentExtraction, R2RDocumentProcessingError], None
        ]

    def __init__(
        self,
        chunking_provider: ChunkingProvider,
        pipe_logger: Optional[RunLoggingSingleton] = None,
        type: PipeType = PipeType.INGESTOR,
        config: Optional[AsyncPipe.PipeConfig] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            pipe_logger=pipe_logger,
            type=type,
            config=config
            or AsyncPipe.PipeConfig(name="default_chunking_pipe"),
            *args,
            **kwargs,
        )
        self.chunking_provider = chunking_provider

    async def _run_logic(
        self,
        input: Input,
        state: AsyncState,
        run_id: Any,
        chunking_config_override: Optional[ChunkingProvider] = None,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[
        Union[R2RDocumentProcessingError, DocumentFragment], None
    ]:
        async for item in input.message:
            if isinstance(item, R2RDocumentProcessingError):
                yield item
                continue

            try:
                iteration = 0
                async for chunk in self.chunking_provider.chunk(item.data):
                    yield DocumentFragment(
                        id=generate_id_from_label(f"{item.id}-{iteration}"),
                        extraction_id=item.id,
                        document_id=item.document_id,
                        user_id=item.user_id,
                        group_ids=item.group_ids,
                        data=chunk,
                        metadata=item.metadata,
                    )
                    iteration += 1
            except Exception as e:
                yield R2RDocumentProcessingError(
                    document_id=item.document_id,
                    error_message=f"Error chunking document: {str(e)}",
                )

import json
import logging
import uuid
from typing import Any, AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.datastructures import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.param_functions import File, Form

from r2r.core import Document, Pipeline, generate_id_from_label

# Current directory where this script is located
MB_CONVERSION_FACTOR = 1024 * 1024


async def list_to_generator(array: list[Any]) -> AsyncGenerator[Any, None]:
    for item in array:
        yield item


class R2RApp:
    def __init__(
        self,
        ingestion_pipeline: Pipeline,
        search_pipeline: Pipeline,
        rag_pipeline: Pipeline,
        do_apply_cors: bool = True,
        *args,
        **kwargs,
    ):
        self.app = FastAPI()
        if do_apply_cors:
            R2RApp._apply_cors(self.app)

        self.ingestion_pipeline = ingestion_pipeline
        self.search_pipeline = search_pipeline
        self.rag_pipeline = rag_pipeline

        self.app.add_api_route(
            path="/ingest_documents/",
            endpoint=self.ingest_documents,
            methods=["POST"],
        )
        self.app.add_api_route(
            path="/ingest_files/", endpoint=self.ingest_files, methods=["POST"]
        )
        self.app.add_api_route(
            path="/search/", endpoint=self.search, methods=["POST"]
        )
        self.app.add_api_route(
            path="/rag/", endpoint=self.rag, methods=["POST"]
        )

    async def ingest_documents(self, documents: list[Document] = []):
        try:
            # Process the documents through the pipeline
            await self.ingestion_pipeline.run(
                input=list_to_generator(documents), pipeline_type="ingestion"
            )
            return {"message": "Entries upserted successfully."}
        except Exception as e:
            logging.error(
                f"ingest_documents(documents={documents}) - \n\n{str(e)})"
            )
            raise HTTPException(status_code=500, detail=str(e))

    async def ingest_files(
        self,
        metadata: str = "{}",
        ids: str = "[]",
        files: list[UploadFile] = [],
    ):
        try:
            ids_list = json.loads(
                ids
            )  #  if ids else []  # Parse the JSON string to a list
            metadata_json = json.loads(metadata)
            documents = []
            for iteration, file in enumerate(files):
                if (
                    file.size
                    > 128  # config.app.get("max_file_size_in_mb", 128)
                    * MB_CONVERSION_FACTOR
                ):
                    raise HTTPException(
                        status_code=413,
                        detail="File size exceeds maximum allowed size.",
                    )

            documents.append(
                Document(
                    id=generate_id_from_label(file.filename)
                    if len(ids_list) == 0
                    else uuid.UUID(ids_list[iteration]),
                    type=file.filename.split(".")[-1],
                    data=await file.read(),
                    metadata=metadata_json,
                )
            )
            # Run the pipeline asynchronously
            await self.ingestion_pipeline.run(
                input=list_to_generator(documents),
                pipeline_type="ingestion",
            )
            return {
                "results": [
                    f"File '{file.filename}' processed successfully for each file"
                    for file in files
                ]
            }
        except Exception as e:
            logging.error(
                f"ingest_files(metadata={metadata}, ids={ids}, files={files}) - \n\n{str(e)})"
            )
            raise HTTPException(status_code=500, detail=str(e))

    async def search(self, query: str):
        try:
            results = await self.search_pipeline.run(
                input=list_to_generator([query])
            )
            return {"results": results}
        except Exception as e:
            logging.error(f"search(query={query}) - \n\n{str(e)})")
            raise HTTPException(status_code=500, detail=str(e))

    async def rag(self, query):
        try:
            results = await self.rag_pipeline.run(
                input=list_to_generator([query])
            )
            return {"results": results}
        except Exception as e:
            logging.error(f"rag(query={query}) - \n\n{str(e)})")
            raise HTTPException(status_code=500, detail=str(e))

    def serve(self, host: str = "0.0.0.0", port: int = 8000):
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "Please install uvicorn using 'pip install uvicorn'"
            )

        uvicorn.run(self.app, host=host, port=port)

    @staticmethod
    def _apply_cors(app):
        # CORS setup
        origins = [
            "*",  # TODO - Change this to the actual frontend URL
            "http://localhost:3000",
            "http://localhost:8000",
        ]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,  # Allows specified origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )

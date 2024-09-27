from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from io import BytesIO
import asyncio
import concurrent.futures
import os
import base64
from unstructured.partition.auto import partition
import logging

logger = logging.getLogger(__name__)

app = FastAPI()


class PartitionRequestModel(BaseModel):
    file_content: bytes
    ingestion_config: Dict


class PartitionResponseModel(BaseModel):
    elements: List[Dict]


executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.environ.get("MAX_INGESTION_WORKERS", 10))
)


def run_partition(file_content: str, ingestion_config: Dict) -> List[Dict]:
    file_content_bytes = base64.b64decode(file_content)
    file_io = BytesIO(file_content_bytes)
    elements = partition(file=file_io, **ingestion_config)
    return [element.to_dict() for element in elements]


@app.get("/health")
async def health_endpoint():
    return {"status": "ok"}


@app.post("/partition", response_model=PartitionResponseModel)
async def partition_endpoint(request: PartitionRequestModel):
    try:
        logger.info(f"Partitioning request received")
        loop = asyncio.get_event_loop()
        elements = await loop.run_in_executor(
            executor,
            run_partition,
            request.file_content,
            request.ingestion_config,
        )
        logger.info(f"Partitioning completed")
        return PartitionResponseModel(elements=elements)
    except Exception as e:
        logger.error(f"Error partitioning file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

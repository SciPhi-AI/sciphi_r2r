import base64
import logging
from pathlib import Path
from typing import Optional
from uuid import UUID

import yaml
from fastapi import Depends, File, Form, UploadFile
from pydantic import Json

from core.base import ChunkingConfig, R2RException
from core.base.api.models.ingestion.responses import WrappedIngestionResponse

from ...main.hatchet import r2r_hatchet
from .base_router import BaseRouter, RunType

logger = logging.getLogger(__name__)


class IngestionRouter(BaseRouter):
    def __init__(self, service, run_type: RunType = RunType.INGESTION):
        super().__init__(service, run_type)
        self.openapi_extras = self.load_openapi_extras()
        self.setup_routes()

    def load_openapi_extras(self):
        yaml_path = (
            Path(__file__).parent / "data" / "ingestion_router_openapi.yml"
        )
        with open(yaml_path, "r") as yaml_file:
            yaml_content = yaml.safe_load(yaml_file)
        return yaml_content

    def setup_routes(self):
        # Note, we use the following verbose input parameters because FastAPI struggles to handle `File` input and `Body` inputs
        # at the same time. Therefore, we must ues `Form` inputs for the metadata, document_ids
        ingest_files_extras = self.openapi_extras.get("ingest_files", {})
        ingest_files_descriptions = ingest_files_extras.get(
            "input_descriptions", {}
        )

        @self.router.post(
            "/ingest_files",
            openapi_extra=ingest_files_extras.get("openapi_extra"),
        )
        @self.base_endpoint
        async def ingest_files_app(
            files: list[UploadFile] = File(
                ..., description=ingest_files_descriptions.get("files")
            ),
            document_ids: Optional[Json[list[UUID]]] = Form(
                None,
                description=ingest_files_descriptions.get("document_ids"),
            ),
            metadatas: Optional[Json[list[dict]]] = Form(
                None, description=ingest_files_descriptions.get("metadatas")
            ),
            chunking_config: Optional[Json[ChunkingConfig]] = Form(
                None,
                description=ingest_files_descriptions.get("chunking_config"),
            ),
            auth_user=Depends(self.service.providers.auth.auth_wrapper),
        ) -> WrappedIngestionResponse:
            """
            Ingest files into the system.

            This endpoint supports multipart/form-data requests, enabling you to ingest files and their associated metadatas into R2R.

            A valid user authentication token is required to access this endpoint, as regular users can only ingest files for their own access. More expansive group permissioning is under development.
            """
            self._validate_chunking_config(chunking_config)
            # Check if the user is a superuser
            is_superuser = auth_user and auth_user.is_superuser

            # Handle user management logic at the request level
            if not auth_user:
                for metadata in metadatas or []:
                    if "user_id" in metadata and (
                        not is_superuser
                        and metadata["user_id"] != str(auth_user.id)
                    ):
                        raise R2RException(
                            status_code=403,
                            message="Non-superusers cannot set user_id in metadata.",
                        )

                # If user is not a superuser, set user_id in metadata
                metadata["user_id"] = str(auth_user.id)

            file_datas = await self._process_files(files)

            workflow_input = {
                "file_data": file_datas[0],
                "document_id": (
                    [str(doc_id) for doc_id in document_ids][0]
                    if document_ids
                    else None
                ),
                "metadata": metadatas[0] if metadatas else None,
                "chunking_config": (
                    chunking_config.json() if chunking_config else None
                ),
                "user": auth_user.json(),
            }

            task_id = r2r_hatchet.client.admin.run_workflow(
                "ingest-file", {"request": workflow_input}
            )

            return {
                "message": f"Ingestion task queued successfully.",
                "task_id": str(task_id),
            }

        update_files_extras = self.openapi_extras.get("update_files", {})
        update_files_descriptions = update_files_extras.get(
            "input_descriptions", {}
        )

        @self.router.post(
            "/update_files",
            openapi_extra=update_files_extras.get("openapi_extra"),
        )
        @self.base_endpoint
        async def update_files_app(
            files: list[UploadFile] = File(
                ..., description=update_files_descriptions.get("files")
            ),
            document_ids: Optional[Json[list[UUID]]] = Form(
                None,
                description=ingest_files_descriptions.get("document_ids"),
            ),
            metadatas: Optional[Json[list[dict]]] = Form(
                None, description=ingest_files_descriptions.get("metadatas")
            ),
            chunking_config: Optional[Json[ChunkingConfig]] = Form(
                None,
                description=ingest_files_descriptions.get("chunking_config"),
            ),
            auth_user=Depends(self.service.providers.auth.auth_wrapper),
        ) -> WrappedIngestionResponse:
            """
            Update existing files in the system.

            This endpoint supports multipart/form-data requests, enabling you to update files and their associated metadatas into R2R.




            A valid user authentication token is required to access this endpoint, as regular users can only update their own files. More expansive group permissioning is under development.
            """

            self._validate_chunking_config(chunking_config)
            # Check if the user is a superuser
            is_superuser = auth_user and auth_user.is_superuser

            # Handle user management logic at the request level
            if not is_superuser:
                for metadata in metadatas or []:
                    if "user_id" in metadata and metadata["user_id"] != str(
                        auth_user.id
                    ):
                        raise R2RException(
                            status_code=403,
                            message="Non-superusers cannot set user_id in metadata.",
                        )

                    # Set user_id in metadata for non-superusers
                    metadata["user_id"] = str(auth_user.id)

            file_datas = await self._process_files(files)

            workflow_input = {
                "file_datas": file_datas,
                "document_ids": [str(doc_id) for doc_id in document_ids],
                "metadatas": metadatas,
                "chunking_config": (
                    chunking_config.json() if chunking_config else None
                ),
                "user": auth_user.json(),
            }

            task_id = r2r_hatchet.client.admin.run_workflow(
                "update-files", {"request": workflow_input}
            )

            return {
                "message": f"Update task queued successfully.",
                "task_id": str(task_id),
            }

    @staticmethod
    def _validate_chunking_config(chunking_config):
        from ..assembly.factory import R2RProviderFactory

        if chunking_config:
            chunking_config.validate()
            R2RProviderFactory.create_chunking_provider(chunking_config)
        else:
            logger.info("No chunking config override provided. Using default.")

    @staticmethod
    async def _process_files(files):
        import base64

        file_datas = []
        for file in files:
            content = await file.read()
            file_datas.append(
                {
                    "filename": file.filename,
                    "content": base64.b64encode(content).decode("utf-8"),
                    "content_type": file.content_type,
                }
            )
        return file_datas

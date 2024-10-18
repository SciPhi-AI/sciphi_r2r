import copy
import logging
import time
from typing import Any, Optional, Union

from sqlalchemy import text

from core.base import VectorEntry, VectorQuantizationType, VectorSearchResult
from core.base.abstractions import VectorSearchSettings
from shared.abstractions.vector import (
    IndexArgsHNSW,
    IndexArgsIVFFlat,
    IndexMeasure,
    IndexMethod,
    VectorTableName,
)

from .base import DatabaseMixin, QueryBuilder
from .vecs.exc import ArgError

logger = logging.getLogger()
from shared.utils import _decorate_vector_type


def index_measure_to_ops(
    measure: IndexMeasure, quantization_type: VectorQuantizationType
):
    return _decorate_vector_type(measure.ops, quantization_type)


class VectorDBMixin(DatabaseMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dimension = kwargs.get("dimension")
        self.quantization_type = kwargs.get("quantization_type")

    async def initialize_vector_db(self):
        # Create the vector table if it doesn't exist
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.project_name}.vectors (
            extraction_id TEXT PRIMARY KEY,
            document_id TEXT,
            user_id TEXT,
            collection_ids TEXT[],
            vector vector({self.dimension}),
            text TEXT,
            metadata JSONB
        );
        CREATE INDEX IF NOT EXISTS idx_vectors_document_id ON {self.project_name}.vectors (document_id);
        CREATE INDEX IF NOT EXISTS idx_vectors_user_id ON {self.project_name}.vectors (user_id);
        CREATE INDEX IF NOT EXISTS idx_vectors_collection_ids ON {self.project_name}.vectors USING GIN (collection_ids);
        CREATE INDEX IF NOT EXISTS idx_vectors_text ON {self.project_name}.vectors USING GIN (to_tsvector('english', text));
        """
        await self.execute_query(query)

    async def upsert(self, entry: VectorEntry) -> None:
        query = f"""
        INSERT INTO {self.project_name}.vectors
        (extraction_id, document_id, user_id, collection_ids, vector, text, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (extraction_id) DO UPDATE
        SET document_id = $2, user_id = $3, collection_ids = $4, vector = $5, text = $6, metadata = $7;
        """
        await self.execute_query(
            query,
            (
                entry.extraction_id,
                entry.document_id,
                entry.user_id,
                entry.collection_ids,
                entry.vector.data,
                entry.text,
                entry.metadata,
            ),
        )

    async def upsert_entries(self, entries: list[VectorEntry]) -> None:
        query = f"""
        INSERT INTO {self.project_name}.vectors
        (extraction_id, document_id, user_id, collection_ids, vector, text, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (extraction_id) DO UPDATE
        SET document_id = $2, user_id = $3, collection_ids = $4, vector = $5, text = $6, metadata = $7;
        """
        params = [
            (
                entry.extraction_id,
                entry.document_id,
                entry.user_id,
                entry.collection_ids,
                entry.vector.data,
                entry.text,
                entry.metadata,
            )
            for entry in entries
        ]
        await self.execute_many(query, params)

    async def semantic_search(
        self, query_vector: list[float], search_settings: VectorSearchSettings
    ) -> list[VectorSearchResult]:
        query = f"""
        SELECT extraction_id, document_id, user_id, collection_ids, text,
               1 - (vector <=> $1::vector) as similarity, metadata
        FROM {self.project_name}.vectors
        WHERE collection_ids && $2
        ORDER BY similarity DESC
        LIMIT $3 OFFSET $4;
        """
        results = await self.fetch_query(
            query,
            (
                query_vector,
                search_settings.collection_ids,
                search_settings.search_limit,
                search_settings.offset,
            ),
        )

        return [
            VectorSearchResult(
                extraction_id=result["extraction_id"],
                document_id=result["document_id"],
                user_id=result["user_id"],
                collection_ids=result["collection_ids"],
                text=result["text"],
                score=float(result["similarity"]),
                metadata=result["metadata"],
            )
            for result in results
        ]

    async def full_text_search(
        self, query_text: str, search_settings: VectorSearchSettings
    ) -> list[VectorSearchResult]:
        query = f"""
        SELECT extraction_id, document_id, user_id, collection_ids, text,
               ts_rank_cd(to_tsvector('english', text), plainto_tsquery('english', $1)) as rank,
               metadata
        FROM {self.project_name}.vectors
        WHERE collection_ids && $2 AND to_tsvector('english', text) @@ plainto_tsquery('english', $1)
        ORDER BY rank DESC
        LIMIT $3 OFFSET $4;
        """
        results = await self.fetch_query(
            query,
            (
                query_text,
                search_settings.collection_ids,
                search_settings.search_limit,
                search_settings.offset,
            ),
        )

        return [
            VectorSearchResult(
                extraction_id=result["extraction_id"],
                document_id=result["document_id"],
                user_id=result["user_id"],
                collection_ids=result["collection_ids"],
                text=result["text"],
                score=float(result["rank"]),
                metadata=result["metadata"],
            )
            for result in results
        ]

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        search_settings: VectorSearchSettings,
        *args,
        **kwargs,
    ) -> list[VectorSearchResult]:
        if search_settings.hybrid_search_settings is None:
            raise ValueError(
                "Please provide a valid `hybrid_search_settings` in the `search_settings`."
            )
        if (
            search_settings.hybrid_search_settings.full_text_limit
            < search_settings.search_limit
        ):
            raise ValueError(
                "The `full_text_limit` must be greater than or equal to the `search_limit`."
            )

        semantic_settings = copy.deepcopy(search_settings)
        semantic_settings.search_limit += search_settings.offset

        full_text_settings = copy.deepcopy(search_settings)
        full_text_settings.hybrid_search_settings.full_text_limit += (
            search_settings.offset
        )

        semantic_results = await self.semantic_search(
            query_vector, semantic_settings
        )
        full_text_results = await self.full_text_search(
            query_text, full_text_settings
        )

        semantic_limit = search_settings.search_limit
        full_text_limit = (
            search_settings.hybrid_search_settings.full_text_limit
        )
        semantic_weight = (
            search_settings.hybrid_search_settings.semantic_weight
        )
        full_text_weight = (
            search_settings.hybrid_search_settings.full_text_weight
        )
        rrf_k = search_settings.hybrid_search_settings.rrf_k

        combined_results = {
            result.extraction_id: {
                "semantic_rank": rank,
                "full_text_rank": full_text_limit,
                "data": result,
            }
            for rank, result in enumerate(semantic_results, 1)
        }

        for rank, result in enumerate(full_text_results, 1):
            if result.extraction_id in combined_results:
                combined_results[result.extraction_id]["full_text_rank"] = rank
            else:
                combined_results[result.extraction_id] = {
                    "semantic_rank": semantic_limit,
                    "full_text_rank": rank,
                    "data": result,
                }

        combined_results = {
            k: v
            for k, v in combined_results.items()
            if v["semantic_rank"] <= semantic_limit * 2
            and v["full_text_rank"] <= full_text_limit * 2
        }

        for result in combined_results.values():
            semantic_score = 1 / (rrf_k + result["semantic_rank"])
            full_text_score = 1 / (rrf_k + result["full_text_rank"])
            result["rrf_score"] = (
                semantic_score * semantic_weight
                + full_text_score * full_text_weight
            ) / (semantic_weight + full_text_weight)

        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )
        offset_results = sorted_results[
            search_settings.offset : search_settings.offset
            + search_settings.search_limit
        ]

        return [
            VectorSearchResult(
                extraction_id=result["data"].extraction_id,
                document_id=result["data"].document_id,
                user_id=result["data"].user_id,
                collection_ids=result["data"].collection_ids,
                text=result["data"].text,
                score=result["rrf_score"],
                metadata={
                    **result["data"].metadata,
                    "semantic_rank": result["semantic_rank"],
                    "full_text_rank": result["full_text_rank"],
                },
            )
            for result in offset_results
        ]

    async def create_index(
        self,
        table_name: Optional[VectorTableName] = None,
        index_method: IndexMethod = IndexMethod.hnsw,
        measure: IndexMeasure = IndexMeasure.cosine_distance,
        index_arguments: Optional[
            Union[IndexArgsHNSW, IndexArgsIVFFlat]
        ] = None,
        index_name: Optional[str] = None,
        concurrently: bool = True,
    ):
        # This method needs to be implemented based on your specific indexing requirements
        pass

    async def delete(
        self, filters: dict[str, Any]
    ) -> dict[str, dict[str, str]]:
        conditions = []
        params = []
        for key, value in filters.items():
            conditions.append(f"{key} = ${len(params) + 1}")
            params.append(value)

        where_clause = " AND ".join(conditions)
        query = f"""
        DELETE FROM {self.project_name}.vectors
        WHERE {where_clause}
        RETURNING extraction_id;
        """
        results = await self.fetch_query(query, params)
        return {
            result["extraction_id"]: {"status": "deleted"}
            for result in results
        }

    async def assign_document_to_collection(
        self, document_id: str, collection_id: str
    ) -> None:
        query = f"""
        UPDATE {self.project_name}.vectors
        SET collection_ids = array_append(collection_ids, $1)
        WHERE document_id = $2 AND NOT ($1 = ANY(collection_ids));
        """
        await self.execute_query(query, (collection_id, document_id))

    async def remove_document_from_collection(
        self, document_id: str, collection_id: str
    ) -> None:
        query = f"""
        UPDATE {self.project_name}.vectors
        SET collection_ids = array_remove(collection_ids, $1)
        WHERE document_id = $2;
        """
        await self.execute_query(query, (collection_id, document_id))

    async def remove_collection_from_documents(
        self, collection_id: str
    ) -> None:
        query = f"""
        UPDATE {self.project_name}.vectors
        SET collection_ids = array_remove(collection_ids, $1)
        WHERE $1 = ANY(collection_ids);
        """
        await self.execute_query(query, (collection_id,))

    async def delete_user(self, user_id: str) -> None:
        query = f"""
        DELETE FROM {self.project_name}.vectors
        WHERE user_id = $1;
        """
        await self.execute_query(query, (user_id,))

    async def delete_collection(self, collection_id: str) -> None:
        query = f"""
        DELETE FROM {self.project_name}.vectors
        WHERE $1 = ANY(collection_ids);
        """
        await self.execute_query(query, (collection_id,))

    async def get_document_chunks(
        self,
        document_id: str,
        offset: int = 0,
        limit: int = -1,
        include_vectors: bool = False,
    ) -> dict[str, Any]:
        vector_select = ", vector" if include_vectors else ""
        limit_clause = f"LIMIT {limit}" if limit > -1 else ""

        query = f"""
        SELECT extraction_id, document_id, user_id, collection_ids, text, metadata
        {vector_select}
        FROM {self.project_name}.vectors
        WHERE document_id = $1
        OFFSET $2
        {limit_clause};
        """
        params = [document_id, offset]
        if limit > -1:
            params.append(limit)

        results = await self.fetch_query(query, params)

        return {
            "chunks": [
                {
                    "extraction_id": result["extraction_id"],
                    "document_id": result["document_id"],
                    "user_id": result["user_id"],
                    "collection_ids": result["collection_ids"],
                    "text": result["text"],
                    "metadata": result["metadata"],
                    **(
                        {"vector": result["vector"]} if include_vectors else {}
                    ),
                }
                for result in results
            ]
        }

    async def create_index(
        self,
        table_name: Optional[VectorTableName] = None,
        measure: IndexMeasure = IndexMeasure.cosine_distance,
        method: IndexMethod = IndexMethod.auto,
        index_arguments: Optional[
            Union[IndexArgsIVFFlat, IndexArgsHNSW]
        ] = None,
        index_name: Optional[str] = None,
        concurrently: bool = True,
    ) -> None:
        """
        Creates an index for the collection.

        Note:
            When `vecs` creates an index on a pgvector column in PostgreSQL, it uses a multi-step
            process that enables performant indexes to be built for large collections with low end
            database hardware.

            Those steps are:

            - Creates a new table with a different name
            - Randomly selects records from the existing table
            - Inserts the random records from the existing table into the new table
            - Creates the requested vector index on the new table
            - Upserts all data from the existing table into the new table
            - Drops the existing table
            - Renames the new table to the existing tables name

            If you create dependencies (like views) on the table that underpins
            a `vecs.Collection` the `create_index` step may require you to drop those dependencies before
            it will succeed.

        Args:
            measure (IndexMeasure, optional): The measure to index for. Defaults to 'cosine_distance'.
            method (IndexMethod, optional): The indexing method to use. Defaults to 'auto'.
            index_arguments: (IndexArgsIVFFlat | IndexArgsHNSW, optional): Index type specific arguments
            replace (bool, optional): Whether to replace the existing index. Defaults to True.
            concurrently (bool, optional): Whether to create the index concurrently. Defaults to True.
        Raises:
            ArgError: If an invalid index method is used, or if *replace* is False and an index already exists.
        """

        if table_name == VectorTableName.CHUNKS:
            table_name = f"{self.client.project_name}.{self.table.name}"
            col_name = "vec"
        elif table_name == VectorTableName.ENTITIES:
            table_name = (
                f"{self.client.project_name}.{VectorTableName.ENTITIES}"
            )
            col_name = "description_embedding"
        elif table_name == VectorTableName.COMMUNITIES:
            table_name = (
                f"{self.client.project_name}.{VectorTableName.COMMUNITIES}"
            )
            col_name = "embedding"
        else:
            raise ArgError("invalid table name")
        if method not in (
            IndexMethod.ivfflat,
            IndexMethod.hnsw,
            IndexMethod.auto,
        ):
            raise ArgError("invalid index method")

        if index_arguments:
            # Disallow case where user submits index arguments but uses the
            # IndexMethod.auto index (index build arguments should only be
            # used with a specific index)
            if method == IndexMethod.auto:
                raise ArgError(
                    "Index build parameters are not allowed when using the IndexMethod.auto index."
                )
            # Disallow case where user specifies one index type but submits
            # index build arguments for the other index type
            if (
                isinstance(index_arguments, IndexArgsHNSW)
                and method != IndexMethod.hnsw
            ) or (
                isinstance(index_arguments, IndexArgsIVFFlat)
                and method != IndexMethod.ivfflat
            ):
                raise ArgError(
                    f"{index_arguments.__class__.__name__} build parameters were supplied but {method} index was specified."
                )

        if method == IndexMethod.auto:
            if self.client._supports_hnsw():
                method = IndexMethod.hnsw
            else:
                method = IndexMethod.ivfflat

        if method == IndexMethod.hnsw and not self.client._supports_hnsw():
            raise ArgError(
                "HNSW Unavailable. Upgrade your pgvector installation to > 0.5.0 to enable HNSW support"
            )

        ops = index_measure_to_ops(
            measure, quantization_type=self.quantization_type
        )

        if ops is None:
            raise ArgError("Unknown index measure")

        concurrently_sql = "CONCURRENTLY" if concurrently else ""

        index_name = (
            index_name or f"ix_{ops}_{method}__{time.strftime('%Y%m%d%H%M%S')}"
        )

        create_index_sql = f"""
        CREATE INDEX {concurrently_sql} {index_name}
        ON {table_name}
        USING {method} ({col_name} {ops}) {self._get_index_options(method, index_arguments)};
        """

        try:
            if concurrently:
                # For concurrent index creation, we need to execute outside a transaction
                await self.execute_query(
                    create_index_sql, isolation_level="AUTOCOMMIT"
                )
            else:
                await self.execute_query(create_index_sql)
        except Exception as e:
            raise Exception(f"Failed to create index: {e}")

        return None

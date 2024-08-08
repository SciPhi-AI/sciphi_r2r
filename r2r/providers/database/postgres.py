import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Literal, Optional, Union
from uuid import UUID

from sqlalchemy import exc, text
from sqlalchemy.engine.url import make_url

from r2r.base import (
    CryptoProvider,
    DatabaseConfig,
    DatabaseProvider,
    DocumentInfo,
    R2RException,
    RelationalDatabaseProvider,
    User,
    UserCreate,
    VectorDatabaseProvider,
    VectorEntry,
    VectorSearchResult,
    generate_id_from_label,
)
from r2r.base.abstractions.user import UserStats

from .vecs import Client, Collection, create_client

logger = logging.getLogger(__name__)


class PostgresVectorDBProvider(VectorDatabaseProvider):
    def __init__(self, config: DatabaseConfig, *args, **kwargs):
        super().__init__(config)
        self.collection: Optional[Collection] = None
        self.vx: Client = kwargs.get("vx", None)
        if not self.vx:
            raise ValueError(
                "Please provide a valid `vx` client to the `PostgresVectorDBProvider`."
            )
        self.collection_name = kwargs.get("collection_name", None)
        if not self.collection_name:
            raise ValueError(
                "Please provide a valid `collection_name` to the `PostgresVectorDBProvider`."
            )
        dimension = kwargs.get("dimension", None)
        if not dimension:
            raise ValueError(
                "Please provide a valid `dimension` to the `PostgresVectorDBProvider`."
            )

        # Check if a complete Postgres URI is provided
        postgres_uri = self.config.extra_fields.get(
            "postgres_uri"
        ) or os.getenv("POSTGRES_URI")

        if postgres_uri:
            # Log loudly that Postgres URI is being used
            logger.warning("=" * 50)
            logger.warning(
                "ATTENTION: Using provided Postgres URI for connection"
            )
            logger.warning("=" * 50)

            # Validate and use the provided URI
            try:
                parsed_uri = make_url(postgres_uri)
                if not all([parsed_uri.username, parsed_uri.database]):
                    raise ValueError(
                        "The provided Postgres URI is missing required components."
                    )
                DB_CONNECTION = postgres_uri

                # Log the sanitized URI (without password)
                sanitized_uri = parsed_uri.set(password="*****")
                logger.info(f"Connecting using URI: {sanitized_uri}")
            except Exception as e:
                raise ValueError(f"Invalid Postgres URI provided: {e}")
        else:
            # Fall back to existing logic for individual connection parameters
            user = self.config.extra_fields.get("user", None) or os.getenv(
                "POSTGRES_USER"
            )
            password = self.config.extra_fields.get(
                "password", None
            ) or os.getenv("POSTGRES_PASSWORD")
            host = self.config.extra_fields.get("host", None) or os.getenv(
                "POSTGRES_HOST"
            )
            port = self.config.extra_fields.get("port", None) or os.getenv(
                "POSTGRES_PORT"
            )
            db_name = self.config.extra_fields.get(
                "db_name", None
            ) or os.getenv("POSTGRES_DBNAME")

            if not all([user, password, host, db_name]):
                raise ValueError(
                    "Error, please set the POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_DBNAME environment variables or provide them in the config."
                )

            # Check if it's a Unix socket connection
            if host.startswith("/") and not port:
                DB_CONNECTION = (
                    f"postgresql://{user}:{password}@/{db_name}?host={host}"
                )
                logger.info("Using Unix socket connection")
            else:
                DB_CONNECTION = (
                    f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
                )
                logger.info("Using TCP connection")

        # The rest of the initialization remains the same
        try:
            self.vx: Client = create_client(DB_CONNECTION)
        except Exception as e:
            raise ValueError(
                f"Error {e} occurred while attempting to connect to the pgvector provider with {DB_CONNECTION}."
            )

        self.collection_name = self.config.extra_fields.get(
            "vecs_collection"
        ) or os.getenv("POSTGRES_VECS_COLLECTION")
        if not self.collection_name:
            raise ValueError(
                "Error, please set a valid POSTGRES_VECS_COLLECTION environment variable or set a 'vecs_collection' in the 'database' settings of your `r2r.toml`."
            )

        self.collection: Optional[Collection] = None
        self._initialize_vector_db(dimension)
        self._create_hybrid_search_function()
        logger.info(
            f"Successfully initialized PGVectorDB with collection: {self.collection_name}"
        )

    def _initialize_vector_db(self, dimension: int) -> None:
        self.collection = self.vx.get_or_create_collection(
            name=self.collection_name, dimension=dimension
        )

    def _create_hybrid_search_function(self):
        hybrid_search_function = f"""
        CREATE OR REPLACE FUNCTION hybrid_search_{self.collection_name}(
            query_text TEXT,
            query_embedding VECTOR(512),
            match_limit INT,
            full_text_weight FLOAT = 1,
            semantic_weight FLOAT = 1,
            rrf_k INT = 50,
            filter_condition JSONB = NULL
        )
        RETURNS SETOF vecs."{self.collection_name}"
        LANGUAGE sql
        AS $$
        WITH full_text AS (
            SELECT
                id,
                ROW_NUMBER() OVER (ORDER BY ts_rank(to_tsvector('english', metadata->>'text'), websearch_to_tsquery(query_text)) DESC) AS rank_ix
            FROM vecs."{self.collection_name}"
            WHERE to_tsvector('english', metadata->>'text') @@ websearch_to_tsquery(query_text)
            AND (filter_condition IS NULL OR (metadata @> filter_condition))
            ORDER BY rank_ix
            LIMIT LEAST(match_limit, 30) * 2
        ),
        semantic AS (
            SELECT
                id,
                ROW_NUMBER() OVER (ORDER BY vec <#> query_embedding) AS rank_ix
            FROM vecs."{self.collection_name}"
            WHERE filter_condition IS NULL OR (metadata @> filter_condition)
            ORDER BY rank_ix
            LIMIT LEAST(match_limit, 30) * 2
        )
        SELECT
            vecs."{self.collection_name}".*
        FROM
            full_text
            FULL OUTER JOIN semantic
                ON full_text.id = semantic.id
            JOIN vecs."{self.collection_name}"
                ON vecs."{self.collection_name}".id = COALESCE(full_text.id, semantic.id)
        ORDER BY
            COALESCE(1.0 / (rrf_k + full_text.rank_ix), 0.0) * full_text_weight +
            COALESCE(1.0 / (rrf_k + semantic.rank_ix), 0.0) * semantic_weight
            DESC
        LIMIT
            LEAST(match_limit, 30);
        $$;
        """
        retry_attempts = 5
        for attempt in range(retry_attempts):
            try:
                with self.vx.Session() as sess:
                    # Acquire an advisory lock
                    sess.execute(text("SELECT pg_advisory_lock(123456789)"))
                    try:
                        sess.execute(text(hybrid_search_function))
                        sess.commit()
                    finally:
                        # Release the advisory lock
                        sess.execute(
                            text("SELECT pg_advisory_unlock(123456789)")
                        )
                break  # Break the loop if successful
            except exc.InternalError as e:
                if "tuple concurrently updated" in str(e):
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    raise  # Re-raise the exception if it's not a concurrency issue
        else:
            raise RuntimeError(
                "Failed to create hybrid search function after multiple attempts"
            )

    def copy(self, entry: VectorEntry, commit=True) -> None:
        if self.collection is None:
            raise ValueError(
                "Please call `initialize_collection` before attempting to run `copy`."
            )

        serializeable_entry = entry.to_serializable()

        self.collection.copy(
            records=[
                (
                    serializeable_entry["id"],
                    serializeable_entry["vector"],
                    serializeable_entry["metadata"],
                )
            ]
        )

    def copy_entries(
        self, entries: list[VectorEntry], commit: bool = True
    ) -> None:
        if self.collection is None:
            raise ValueError(
                "Please call `initialize_collection` before attempting to run `copy_entries`."
            )

        self.collection.copy(
            records=[
                (
                    str(entry.id),
                    entry.vector.data,
                    entry.to_serializable()["metadata"],
                )
                for entry in entries
            ]
        )

    def upsert(self, entry: VectorEntry, commit=True) -> None:
        if self.collection is None:
            raise ValueError(
                "Please call `initialize_collection` before attempting to run `upsert`."
            )

        self.collection.upsert(
            records=[
                (
                    str(entry.id),
                    entry.vector.data,
                    entry.to_serializable()["metadata"],
                )
            ]
        )

    def upsert_entries(
        self, entries: list[VectorEntry], commit: bool = True
    ) -> None:
        if self.collection is None:
            raise ValueError(
                "Please call `initialize_collection` before attempting to run `upsert_entries`."
            )

        self.collection.upsert(
            records=[
                (
                    str(entry.id),
                    entry.vector.data,
                    entry.to_serializable()["metadata"],
                )
                for entry in entries
            ]
        )

    def search(
        self,
        query_vector: list[float],
        filters: dict[str, Union[bool, int, str]] = {},
        limit: int = 10,
        *args,
        **kwargs,
    ) -> list[VectorSearchResult]:
        if self.collection is None:
            raise ValueError(
                "Please call `initialize_collection` before attempting to run `search`."
            )
        measure = kwargs.get("measure", "cosine_distance")
        mapped_filters = {
            key: {"$eq": value} for key, value in filters.items()
        }

        return [
            VectorSearchResult(id=ele[0], score=float(1 - ele[1]), metadata=ele[2])  # type: ignore
            for ele in self.collection.query(
                data=query_vector,
                limit=limit,
                filters=mapped_filters,
                measure=measure,
                include_value=True,
                include_metadata=True,
            )
        ]

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        limit: int = 10,
        filters: Optional[dict[str, Union[bool, int, str]]] = None,
        full_text_weight: float = 1.0,
        semantic_weight: float = 1.0,
        rrf_k: int = 20,
        *args,
        **kwargs,
    ) -> list[VectorSearchResult]:
        if self.collection is None:
            raise ValueError(
                "Please call `initialize_collection` before attempting to run `hybrid_search`."
            )

        # Convert filters to a JSON-compatible format
        filter_condition = None
        if filters:
            filter_condition = json.dumps(filters)

        query = text(
            f"""
            SELECT * FROM hybrid_search_{self.collection_name}(
                cast(:query_text as TEXT), cast(:query_embedding as VECTOR), cast(:match_limit as INT),
                cast(:full_text_weight as FLOAT), cast(:semantic_weight as FLOAT), cast(:rrf_k as INT),
                cast(:filter_condition as JSONB)
            )
        """
        )

        params = {
            "query_text": str(query_text),
            "query_embedding": list(query_vector),
            "match_limit": limit,
            "full_text_weight": full_text_weight,
            "semantic_weight": semantic_weight,
            "rrf_k": rrf_k,
            "filter_condition": filter_condition,
        }

        with self.vx.Session() as session:
            result = session.execute(query, params).fetchall()
        return [
            VectorSearchResult(id=row[0], score=1.0, metadata=row[-1])
            for row in result
        ]

    def create_index(self, index_type, column_name, index_options):
        self.collection.create_index()

    def delete_by_metadata(
        self,
        metadata_fields: list[str],
        metadata_values: list[Union[bool, int, str]],
        logic: Literal["AND", "OR"] = "AND",
    ) -> list[str]:
        if logic == "OR":
            raise ValueError(
                "OR logic is still being tested before official support for `delete_by_metadata` in pgvector."
            )
        if self.collection is None:
            raise ValueError(
                "Please call `initialize_collection` before attempting to run `delete_by_metadata`."
            )

        if len(metadata_fields) != len(metadata_values):
            raise ValueError(
                "The number of metadata fields must match the number of metadata values."
            )

        # Construct the filter
        if logic == "AND":
            filters = {
                k: {"$eq": v} for k, v in zip(metadata_fields, metadata_values)
            }
        else:  # OR logic
            # TODO - Test 'or' logic and remove check above
            filters = {
                "$or": [
                    {k: {"$eq": v}}
                    for k, v in zip(metadata_fields, metadata_values)
                ]
            }
        return self.collection.delete(filters=filters)

    def get_metadatas(
        self,
        metadata_fields: list[str],
        filter_field: Optional[str] = None,
        filter_value: Optional[Union[bool, int, str]] = None,
    ) -> list[dict]:
        if self.collection is None:
            raise ValueError(
                "Please call `initialize_collection` before attempting to run `get_metadatas`."
            )

        results = {tuple(metadata_fields): {}}
        for field in metadata_fields:
            unique_values = self.collection.get_unique_metadata_values(
                field=field,
                filter_field=filter_field,
                filter_value=filter_value,
            )
            for value in unique_values:
                if value not in results:
                    results[value] = {}
                results[value][field] = value

        return [
            results[key] for key in results if key != tuple(metadata_fields)
        ]

    def get_document_chunks(self, document_id: str) -> list[dict]:
        if not self.collection:
            raise ValueError("Collection is not initialized.")

        table_name = self.collection.table.name
        query = text(
            f"""
            SELECT metadata
            FROM vecs."{table_name}"
            WHERE metadata->>'document_id' = :document_id
            ORDER BY CAST(metadata->>'chunk_order' AS INTEGER)
        """
        )

        params = {"document_id": document_id}

        with self.vx.Session() as sess:
            results = sess.execute(query, params).fetchall()
            return [result[0] for result in results]


class PostgresRelationalDBProvider(RelationalDatabaseProvider):
    def __init__(
        self,
        config: DatabaseConfig,
        crypto_provider: Optional[CryptoProvider],
        *args,
        **kwargs,
    ):
        super().__init__(config)
        self.vx: Client = kwargs.get("vx", None)
        self.crypto_provider = crypto_provider
        if not self.vx:
            raise ValueError(
                "Please provide a valid `vx` client to the `PostgresRelationalDBProvider`."
            )
        self.collection_name = kwargs.get("collection_name", None)
        if not self.collection_name:
            raise ValueError(
                "Please provide a valid `collection_name` to the `PostgresRelationalDBProvider`."
            )
        self._initialize_relational_db()

    def _initialize_relational_db(self):
        with self.vx.Session() as sess:
            with sess.begin():
                try:
                    # Enable uuid-ossp extension
                    sess.execute(
                        text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
                    )
                except exc.ProgrammingError as e:
                    logger.error(f"Error enabling uuid-ossp extension: {e}")
                    raise

                # Create the document info table if it doesn't exist
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS document_info_{self.collection_name} (
                    document_id UUID PRIMARY KEY,
                    title TEXT,
                    user_id UUID NULL,
                    group_ids UUID[] NULL,
                    version TEXT,
                    size_in_bytes INT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    metadata JSONB,
                    status TEXT DEFAULT 'processing'
                );
                """
                sess.execute(text(create_table_query))

                # Create the users table if it doesn't exist
                query = f"""
                CREATE TABLE IF NOT EXISTS users_{self.collection_name} (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    is_superuser BOOLEAN DEFAULT FALSE,
                    is_active BOOLEAN DEFAULT TRUE,
                    is_verified BOOLEAN DEFAULT FALSE,
                    verification_code TEXT,
                    verification_code_expiry TIMESTAMPTZ,
                    name TEXT,
                    bio TEXT,
                    profile_picture TEXT,
                    reset_token TEXT,
                    reset_token_expiry TIMESTAMPTZ,
                    group_ids UUID[] NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
                sess.execute(text(query))

                # Create blacklisted tokens table
                query = f"""
                CREATE TABLE IF NOT EXISTS blacklisted_tokens_{self.collection_name} (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    token TEXT NOT NULL,
                    blacklisted_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_blacklisted_tokens_{self.collection_name}_token
                ON blacklisted_tokens_{self.collection_name} (token);
                CREATE INDEX IF NOT EXISTS idx_blacklisted_tokens_{self.collection_name}_blacklisted_at
                ON blacklisted_tokens_{self.collection_name} (blacklisted_at);
                """
                sess.execute(text(query))

                # Create groups table
                query = f"""
                CREATE TABLE IF NOT EXISTS groups_{self.collection_name} (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
                sess.execute(text(query))

                # Create the index on group_ids
                create_index_query = f"""
                CREATE INDEX IF NOT EXISTS idx_group_ids_{self.collection_name}
                ON document_info_{self.collection_name} USING GIN (group_ids);
                """
                sess.execute(text(create_index_query))

                sess.commit()

    def upsert_documents_overview(
        self, documents_overview: list[DocumentInfo]
    ) -> None:
        for document_info in documents_overview:
            db_entry = document_info.convert_to_db_entry()

            # Convert 'None' string to None type for user_id
            if db_entry["user_id"] == "None":
                db_entry["user_id"] = None

            if (
                db_entry["group_ids"] == "None"
                or db_entry["group_ids"] == "[]"
            ):
                db_entry["group_ids"] = None
            elif isinstance(db_entry["group_ids"], list):
                # ensure group_ids are strings
                db_entry["group_ids"] = [
                    str(gid) for gid in db_entry["group_ids"]
                ]

            query = text(
                f"""
                INSERT INTO document_info_{self.collection_name}
                (document_id, title, user_id, group_ids, version, created_at, updated_at, size_in_bytes, metadata, status)
                VALUES (:document_id, :title, :user_id, :group_ids, :version, :created_at, :updated_at, :size_in_bytes, :metadata, :status)
                ON CONFLICT (document_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    user_id = EXCLUDED.user_id,
                    group_ids = EXCLUDED.group_ids,
                    version = EXCLUDED.version,
                    updated_at = EXCLUDED.updated_at,
                    size_in_bytes = EXCLUDED.size_in_bytes,
                    metadata = EXCLUDED.metadata,
                    status = EXCLUDED.status;
            """
            )
            with self.vx.Session() as sess:
                sess.execute(query, db_entry)
                sess.commit()

    def delete_from_documents_overview(
        self, document_id: str, version: Optional[str] = None
    ) -> None:
        query = f"""
            DELETE FROM document_info_{self.collection_name}
            WHERE document_id = :document_id
        """
        params = {"document_id": document_id}

        if version is not None:
            query += " AND version = :version"
            params["version"] = version

        with self.vx.Session() as sess:
            with sess.begin():
                sess.execute(text(query), params)
            sess.commit()

    def get_documents_overview(
        self,
        filter_document_ids: Optional[list[str]] = None,
        filter_user_ids: Optional[list[str]] = None,
        filter_group_ids: Optional[list[str]] = None,
    ):
        conditions = []
        params = {}

        if filter_document_ids:
            placeholders = ", ".join(
                f":doc_id_{i}" for i in range(len(filter_document_ids))
            )
            conditions.append(f"document_id IN ({placeholders})")
            params.update(
                {
                    f"doc_id_{i}": str(document_id)
                    for i, document_id in enumerate(filter_document_ids)
                }
            )

        if filter_user_ids:
            placeholders = ", ".join(
                f":user_id_{i}" for i in range(len(filter_user_ids))
            )
            conditions.append(f"user_id IN ({placeholders})")
            params.update(
                {
                    f"user_id_{i}": str(user_id)
                    for i, user_id in enumerate(filter_user_ids)
                }
            )

        if filter_group_ids:
            group_conditions = []
            for i, group_id in enumerate(filter_group_ids):
                group_conditions.append(f":group_id_{i} = ANY(group_ids)")
                params[f"group_id_{i}"] = str(group_id)
            conditions.append(f"({' OR '.join(group_conditions)})")

        query = f"""
            SELECT document_id, title, user_id, group_ids, version, size_in_bytes, created_at, updated_at, metadata, status
            FROM document_info_{self.collection_name}
        """
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self.vx.Session() as sess:
            results = sess.execute(text(query), params).fetchall()
            return [
                DocumentInfo(
                    document_id=row[0],
                    title=row[1],
                    user_id=row[2],
                    group_ids=row[3],
                    version=row[4],
                    size_in_bytes=row[5],
                    created_at=row[6],
                    updated_at=row[7],
                    metadata=row[8],
                    status=row[9],
                )
                for row in results
            ]

    # TODO - Deprecate this method
    def get_users_overview(self, user_ids: Optional[list[str]] = None):
        user_ids_condition = ""
        params = {}
        if user_ids:
            user_ids_condition = "WHERE user_id IN :user_ids"
            params["user_ids"] = tuple(
                map(str, user_ids)
            )  # Convert UUIDs to strings

        query = f"""
            SELECT user_id, COUNT(document_id) AS num_files, SUM(size_in_bytes) AS total_size_in_bytes, ARRAY_AGG(document_id) AS document_ids
            FROM document_info_{self.collection_name}
            {user_ids_condition}
            GROUP BY user_id
        """

        with self.vx.Session() as sess:
            results = sess.execute(text(query), params).fetchall()
        return [
            UserStats(
                user_id=row[0],
                num_files=row[1],
                total_size_in_bytes=row[2],
                document_ids=row[3],
            )
            for row in results
            if row[0] is not None
        ]

    # Group management methods
    def create_group(self, name: str, description: str = "") -> dict:
        current_time = datetime.utcnow()
        query = text(
            f"""
            INSERT INTO groups_{self.collection_name} (name, description, created_at, updated_at)
            VALUES (:name, :description, :created_at, :updated_at)
            RETURNING id, name, description, created_at, updated_at
            """
        )
        try:
            with self.vx.Session() as sess:
                result = sess.execute(
                    query,
                    {
                        "name": name,
                        "description": description,
                        "created_at": current_time,
                        "updated_at": current_time,
                    },
                )
                group_data = result.fetchone()
                sess.commit()

            return {
                "id": group_data[0],
                "name": group_data[1],
                "description": group_data[2],
                "created_at": group_data[3],
                "updated_at": group_data[4],
            }
        except Exception as e:
            logger.error(f"Error creating group: {e}")
            raise

    def get_group(self, group_id: UUID) -> Optional[dict]:
        query = text(
            f"""
            SELECT id, name, description, created_at, updated_at
            FROM groups_{self.collection_name}
            WHERE id = :group_id
            """
        )
        with self.vx.Session() as sess:
            result = sess.execute(query, {"group_id": group_id})
            group_data = result.fetchone()
        return dict(zip(result.keys(), group_data)) if group_data else None

    def get_group_count(self) -> int:
        query = text(
            f"""
            SELECT COUNT(*) FROM groups_{self.collection_name}
            """
        )
        with self.vx.Session() as sess:
            result = sess.execute(query)
            count = result.scalar()
        return count

    def update_group(
        self, group_id: UUID, name: str = None, description: str = None
    ) -> bool:
        print("in vector update ....")
        update_fields = []
        params = {"group_id": group_id}
        if name is not None:
            update_fields.append("name = :name")
            params["name"] = name
        if description is not None:
            update_fields.append("description = :description")
            params["description"] = description

        if not update_fields:
            return False

        query = text(
            f"""
            UPDATE groups_{self.collection_name}
            SET {", ".join(update_fields)}, updated_at = NOW()
            WHERE id = :group_id
            """
        )
        with self.vx.Session() as sess:
            result = sess.execute(query, params)
            sess.commit()
        return result.rowcount > 0

    def delete_group(self, group_id: UUID) -> bool:
        with self.vx.Session() as sess:
            try:
                # Start a transaction
                sess.begin()

                # Delete the group
                delete_group_query = text(
                    f"""
                    DELETE FROM groups_{self.collection_name}
                    WHERE id = :group_id
                    """
                )
                result = sess.execute(
                    delete_group_query, {"group_id": group_id}
                )

                if result.rowcount == 0:
                    # Group not found, rollback and return False
                    sess.rollback()
                    return False

                # Update users' group_ids
                update_users_query = text(
                    f"""
                    UPDATE users_{self.collection_name}
                    SET group_ids = array_remove(group_ids, :group_id)
                    WHERE :group_id = ANY(group_ids)
                    """
                )
                sess.execute(update_users_query, {"group_id": group_id})

                # Update documents' group_ids
                update_documents_query = text(
                    f"""
                    UPDATE document_info_{self.collection_name}
                    SET group_ids = array_remove(group_ids, :group_id)
                    WHERE :group_id = ANY(group_ids)
                    """
                )
                sess.execute(update_documents_query, {"group_id": group_id})

                # Commit the transaction
                sess.commit()

                return True

            except Exception as e:
                # If any error occurs, rollback the transaction
                sess.rollback()
                logger.error(f"Error deleting group: {e}")
                return False

    def list_groups(self, offset: int = 0, limit: int = 100) -> list[dict]:
        query = text(
            f"""
            SELECT id, name, description, created_at, updated_at
            FROM groups_{self.collection_name}
            ORDER BY name
            OFFSET :offset
            LIMIT :limit
            """
        )
        with self.vx.Session() as sess:
            result = sess.execute(query, {"offset": offset, "limit": limit})
            columns = result.keys()
            groups = result.fetchall()
        return [dict(zip(columns, group)) for group in groups]

    # User-Group management methods
    def add_user_to_group(self, user_id: UUID, group_id: UUID) -> bool:
        query = text(
            f"""
            UPDATE users_{self.collection_name}
            SET group_ids = array_append(group_ids, :group_id)
            WHERE id = :user_id AND NOT (:group_id = ANY(group_ids))
            RETURNING id
            """
        )
        with self.vx.Session() as sess:
            result = sess.execute(
                query, {"user_id": user_id, "group_id": group_id}
            )
            updated = result.fetchone() is not None
            if updated:
                sess.commit()
                return updated

    def remove_user_from_group(self, user_id: UUID, group_id: UUID) -> bool:
        query = text(
            f"""
            UPDATE users_{self.collection_name}
            SET group_ids = array_remove(group_ids, :group_id)
            WHERE id = :user_id AND :group_id = ANY(group_ids)
            RETURNING id
            """
        )
        with self.vx.Session() as sess:
            result = sess.execute(
                query, {"user_id": user_id, "group_id": group_id}
            )
            updated = result.fetchone() is not None
            if updated:
                sess.commit()
            return updated

    def get_users_in_group(
        self, group_id: UUID, offset: int = 0, limit: int = 100
    ) -> list[User]:
        query = text(
            f"""
            SELECT id, email, is_superuser, is_active, is_verified, created_at, updated_at, group_ids
            FROM users_{self.collection_name}
            WHERE :group_id = ANY(group_ids)
            ORDER BY email
            OFFSET :offset
            LIMIT :limit
            """
        )
        with self.vx.Session() as sess:
            result = sess.execute(
                query, {"group_id": group_id, "offset": offset, "limit": limit}
            )
            users_data = result.fetchall()
        return [
            User(
                id=user_data[0],
                email=user_data[1],
                hashed_password="null",
                is_superuser=user_data[2],
                is_active=user_data[3],
                is_verified=user_data[4],
                created_at=user_data[5],
                updated_at=user_data[6],
                group_ids=user_data[7],
            )
            for user_data in users_data
        ]

    def get_groups_for_user(self, user_id: UUID) -> list[dict]:
        query = text(
            f"""
            SELECT g.id, g.name, g.description, g.created_at, g.updated_at
            FROM groups_{self.collection_name} g
            JOIN users_{self.collection_name} u ON g.id = ANY(u.group_ids)
            WHERE u.id = :user_id
            ORDER BY g.name
            """
        )
        with self.vx.Session() as sess:
            result = sess.execute(query, {"user_id": user_id})
            columns = result.keys()
            groups = result.fetchall()
        return [dict(zip(columns, group)) for group in groups]

    def create_user(self, user: UserCreate) -> User:
        hashed_password = self.crypto_provider.get_password_hash(user.password)
        query = text(
            f"""
            INSERT INTO users_{self.collection_name}
            (email, id, hashed_password, group_ids)
            VALUES (:email, :id, :hashed_password, :group_ids)
            RETURNING id, email, is_superuser, is_active, is_verified, created_at, updated_at, group_ids
            """
        )

        with self.vx.Session() as sess:
            result = sess.execute(
                query,
                {
                    "email": user.email,
                    "id": generate_id_from_label(user.email),
                    "hashed_password": hashed_password,
                    "group_ids": [],
                },
            )
            user_data = result.fetchone()
            sess.commit()

        return User(
            id=user_data[0],
            email=user_data[1],
            is_superuser=user_data[2],
            is_active=user_data[3],
            is_verified=user_data[4],
            created_at=user_data[5],
            updated_at=user_data[6],
            group_ids=user_data[7],
            hashed_password=hashed_password,
        )

    def get_user_by_email(self, email: str) -> Optional[User]:
        query = text(
            f"""
            SELECT id, email, hashed_password, is_superuser, is_active, is_verified, created_at, updated_at, group_ids
            FROM users_{self.collection_name}
            WHERE email = :email
            """
        )

        with self.vx.Session() as sess:
            result = sess.execute(query, {"email": email})
            user_data = result.fetchone()

        if user_data:
            return User(
                id=user_data[0],
                email=user_data[1],
                hashed_password=user_data[2],
                is_superuser=user_data[3],
                is_active=user_data[4],
                is_verified=user_data[5],
                created_at=user_data[6],
                updated_at=user_data[7],
                group_ids=user_data[8],
            )
        return None

    def store_verification_code(
        self, user_id: UUID, verification_code: str, expiry: datetime
    ):
        query = text(
            f"""
        UPDATE users_{self.collection_name}
        SET verification_code = :code, verification_code_expiry = :expiry
        WHERE id = :user_id
        """
        )

        with self.vx.Session() as sess:
            sess.execute(
                query,
                {
                    "code": verification_code,
                    "expiry": expiry,
                    "user_id": user_id,
                },
            )
            sess.commit()

    def get_user_id_by_verification_code(
        self, verification_code: str
    ) -> Optional[UUID]:
        query = text(
            f"""
        SELECT id FROM users_{self.collection_name}
        WHERE verification_code = :code AND verification_code_expiry > NOW()
        """
        )

        with self.vx.Session() as sess:
            result = sess.execute(query, {"code": verification_code})
            user_data = result.fetchone()

        return user_data[0] if user_data else None

    def mark_user_as_verified(self, user_id: UUID):
        query = text(
            f"""
        UPDATE users_{self.collection_name}
        SET is_verified = TRUE, verification_code = NULL, verification_code_expiry = NULL
        WHERE id = :user_id
        """
        )

        with self.vx.Session() as sess:
            sess.execute(query, {"user_id": user_id})
            sess.commit()

    def mark_user_as_superuser(self, user_id: UUID):
        query = text(
            f"""
        UPDATE users_{self.collection_name}
        SET is_superuser = TRUE, is_verified = TRUE, verification_code = NULL, verification_code_expiry = NULL
        WHERE id = :user_id
        """
        )

        with self.vx.Session() as sess:
            sess.execute(query, {"user_id": user_id})
            sess.commit()

    def remove_verification_code(self, verification_code: str):
        query = text(
            f"""
        UPDATE users_{self.collection_name}
        SET verification_code = NULL, verification_code_expiry = NULL
        WHERE verification_code = :code
        """
        )

        with self.vx.Session() as sess:
            sess.execute(query, {"code": verification_code})
            sess.commit()

    def remove_user_from_all_groups(self, user_id: UUID):
        query = text(
            f"""
            UPDATE users_{self.collection_name}
            SET group_ids = ARRAY[]::UUID[]
            WHERE id = :user_id
            """
        )
        with self.vx.Session() as sess:
            sess.execute(query, {"user_id": user_id})
            sess.commit()

    def handle_user_documents(self, user_id: UUID):
        # For now, we'll just delete the user's documents
        # In the future, you might want to implement a transfer ownership feature
        query = text(
            f"""
            DELETE FROM document_info_{self.collection_name}
            WHERE user_id = :user_id
            """
        )
        with self.vx.Session() as sess:
            sess.execute(query, {"user_id": user_id})
            sess.commit()

    def invalidate_user_tokens(self, user_id: UUID):
        # This method should blacklist all tokens for the user
        # For simplicity, we'll just delete all tokens for the user
        query = text(
            f"""
            DELETE FROM blacklisted_tokens_{self.collection_name}
            WHERE token IN (
                SELECT token
                FROM users_{self.collection_name}
                WHERE id = :user_id
            )
            """
        )
        with self.vx.Session() as sess:
            sess.execute(query, {"user_id": user_id})
            sess.commit()

    def delete_user(self, user_id: UUID):
        with self.vx.Session() as sess:
            try:
                sess.begin()

                # Remove user from groups
                self.remove_user_from_all_groups(user_id)

                # Handle user's documents
                self.handle_user_documents(user_id)

                # Delete user
                result = sess.execute(
                    text(
                        f"DELETE FROM users_{self.collection_name} WHERE id = :user_id"
                    ),
                    {"user_id": user_id},
                )

                if result.rowcount == 0:
                    raise R2RException(
                        status_code=404, message="User not found"
                    )

                sess.commit()

                # Invalidate user's tokens
                self.invalidate_user_tokens(user_id)
            except Exception as e:
                sess.rollback()
                logger.error(f"Error deleting user: {e}")
                raise R2RException(
                    status_code=500, message="Failed to delete user"
                )

    def delete_document(self, document_id: str):
        with self.vx.Session() as sess:
            try:
                sess.begin()

                # Delete from document_info
                sess.execute(
                    text(
                        f"DELETE FROM document_info_{self.collection_name} WHERE document_id = :doc_id"
                    ),
                    {"doc_id": document_id},
                )

                # Delete from vector database
                self.vector_db.delete(document_id)

                # Update user statistics
                self.update_user_document_stats(document_id)

                sess.commit()
            except Exception as e:
                sess.rollback()
                logger.error(f"Error deleting document: {e}")

    def get_all_users(self) -> list[User]:
        query = text(
            f"""
            SELECT id, email, is_superuser, is_active, is_verified, created_at, updated_at, group_ids
            FROM users_{self.collection_name}
            """
        )

        with self.vx.Session() as sess:
            result = sess.execute(query)
            users_data = result.fetchall()

        return [
            User(
                id=user_data[0],
                email=user_data[1],
                hashed_password="null",
                is_superuser=user_data[2],
                is_active=user_data[3],
                is_verified=user_data[4],
                created_at=user_data[5],
                updated_at=user_data[6],
                group_ids=user_data[7],
            )
            for user_data in users_data
        ]

    def expire_verification_code(self, user_id):
        query = text(
            f"""
        UPDATE users_{self.collection_name}
        SET verification_code_expiry = NOW() - INTERVAL '365 day'
        WHERE id = :user_id
        """
        )
        with self.vx.Session() as sess:
            sess.execute(query, {"user_id": user_id})
            sess.commit()

    def store_reset_token(
        self, user_id: UUID, reset_token: str, expiry: datetime
    ):
        query = text(
            f"""
            UPDATE users_{self.collection_name}
            SET reset_token = :token, reset_token_expiry = :expiry
            WHERE id = :user_id
            """
        )

        with self.vx.Session() as sess:
            sess.execute(
                query,
                {
                    "token": reset_token,
                    "expiry": expiry,
                    "user_id": user_id,
                },
            )
            sess.commit()

    def get_user_id_by_reset_token(self, reset_token: str) -> Optional[UUID]:
        query = text(
            f"""
            SELECT id FROM users_{self.collection_name}
            WHERE reset_token = :token AND reset_token_expiry > NOW()
            """
        )

        with self.vx.Session() as sess:
            result = sess.execute(query, {"token": reset_token})
            user_data = result.fetchone()

        return user_data[0] if user_data else None

    def remove_reset_token(self, user_id: UUID):
        query = text(
            f"""
            UPDATE users_{self.collection_name}
            SET reset_token = NULL, reset_token_expiry = NULL
            WHERE id = :user_id
            """
        )

        with self.vx.Session() as sess:
            sess.execute(query, {"user_id": user_id})
            sess.commit()

    def blacklist_token(self, token: str, current_time: datetime = None):
        if current_time is None:
            current_time = datetime.utcnow()
        query = text(
            f"""
            INSERT INTO blacklisted_tokens_{self.collection_name} (token, blacklisted_at)
            VALUES (:token, :blacklisted_at)
            """
        )

        with self.vx.Session() as sess:
            sess.execute(
                query, {"token": token, "blacklisted_at": current_time}
            )
            sess.commit()

    def is_token_blacklisted(self, token: str) -> bool:
        query = text(
            f"""
            SELECT EXISTS(
                SELECT 1 FROM blacklisted_tokens_{self.collection_name}
                WHERE token = :token
            )
            """
        )

        with self.vx.Session() as sess:
            result = sess.execute(query, {"token": token})
            return result.scalar()

    def clean_expired_blacklisted_tokens(
        self,
        max_age_hours: int = 7 * 24,
        current_time: Optional[datetime] = None,
    ):
        if current_time is None:
            current_time = datetime.utcnow()
        expiry_time = current_time - timedelta(hours=max_age_hours)
        query = text(
            f"""
            DELETE FROM blacklisted_tokens_{self.collection_name}
            WHERE blacklisted_at < :expiry_time
            """
        )

        with self.vx.Session() as sess:
            sess.execute(query, {"expiry_time": expiry_time})
            sess.commit()

    # Modify existing methods to include new profile fields
    def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        query = text(
            f"""
            SELECT id, email, hashed_password, is_superuser, is_active, is_verified,
                   created_at, updated_at, name, profile_picture, bio, group_ids
            FROM users_{self.collection_name}
            WHERE id = :user_id
            """
        )

        with self.vx.Session() as sess:
            result = sess.execute(query, {"user_id": user_id})
            user_data = result.fetchone()

        if user_data:
            return User(
                id=user_data[0],
                email=user_data[1],
                hashed_password=user_data[2],
                is_superuser=user_data[3],
                is_active=user_data[4],
                is_verified=user_data[5],
                created_at=user_data[6],
                updated_at=user_data[7],
                name=user_data[8],
                profile_picture=user_data[9],
                bio=user_data[10],
                group_ids=user_data[11],
            )
        return None

    def update_user(self, user: User) -> User:
        query = text(
            f"""
            UPDATE users_{self.collection_name}
            SET email = :email, is_superuser = :is_superuser, is_active = :is_active,
                is_verified = :is_verified, updated_at = NOW(), name = :name,
                profile_picture = :profile_picture, bio = :bio, group_ids = :group_ids
            WHERE id = :user_id
            RETURNING id, email, is_superuser, is_active, is_verified, created_at,
                      updated_at, name, profile_picture, bio, group_ids
            """
        )

        with self.vx.Session() as sess:
            result = sess.execute(
                query,
                {
                    "email": user.email,
                    "is_superuser": user.is_superuser,
                    "is_active": user.is_active,
                    "is_verified": user.is_verified,
                    "user_id": user.id,
                    "name": user.name,
                    "profile_picture": user.profile_picture,
                    "bio": user.bio,
                    "group_ids": user.group_ids,
                },
            )
            updated_user_data = result.fetchone()
            sess.commit()

        return User(
            id=updated_user_data[0],
            email=updated_user_data[1],
            hashed_password="null",
            is_superuser=updated_user_data[2],
            is_active=updated_user_data[3],
            is_verified=updated_user_data[4],
            created_at=updated_user_data[5],
            updated_at=updated_user_data[6],
            name=updated_user_data[7],
            profile_picture=updated_user_data[8],
            bio=updated_user_data[9],
            group_ids=updated_user_data[10],
        )

    def update_user_password(self, user_id: UUID, new_hashed_password: str):
        query = text(
            f"""
            UPDATE users_{self.collection_name}
            SET hashed_password = :new_hashed_password, updated_at = NOW()
            WHERE id = :user_id
            """
        )

        with self.vx.Session() as sess:
            sess.execute(
                query,
                {
                    "user_id": user_id,
                    "new_hashed_password": new_hashed_password,
                },
            )
            sess.commit()


class PostgresDBProvider(DatabaseProvider):
    def __init__(
        self,
        config: DatabaseConfig,
        dimension: int,
        crypto_provider: Optional[CryptoProvider] = None,
        *args,
        **kwargs,
    ):
        user = config.extra_fields.get("user", None) or os.getenv(
            "POSTGRES_USER"
        )
        if not user:
            raise ValueError(
                "Error, please set a valid POSTGRES_USER environment variable or set a 'user' in the 'database' settings of your `r2r.toml`."
            )
        password = config.extra_fields.get("password", None) or os.getenv(
            "POSTGRES_PASSWORD"
        )
        if not password:
            raise ValueError(
                "Error, please set a valid POSTGRES_PASSWORD environment variable or set a 'password' in the 'database' settings of your `r2r.toml`."
            )

        host = config.extra_fields.get("host", None) or os.getenv(
            "POSTGRES_HOST"
        )
        if not host:
            raise ValueError(
                "Error, please set a valid POSTGRES_HOST environment variable or set a 'host' in the 'database' settings of your `r2r.toml`."
            )

        port = config.extra_fields.get("port", None) or os.getenv(
            "POSTGRES_PORT"
        )
        if not port:
            raise ValueError(
                "Error, please set a valid POSTGRES_PORT environment variable or set a 'port' in the 'database' settings of your `r2r.toml`."
            )

        db_name = config.extra_fields.get("db_name", None) or os.getenv(
            "POSTGRES_DBNAME"
        )
        if not db_name:
            raise ValueError(
                "Error, please set a valid POSTGRES_DBNAME environment variable or set a 'db_name' in the 'database' settings of your `r2r.toml`."
            )

        collection_name = config.extra_fields.get(
            "vecs_collection", None
        ) or os.getenv("POSTGRES_VECS_COLLECTION")
        if not collection_name:
            raise ValueError(
                "Error, please set a valid POSTGRES_VECS_COLLECTION environment variable or set a 'collection' in the 'database' settings of your `r2r.toml`."
            )

        if not all([user, password, host, port, db_name, collection_name]):
            raise ValueError(
                "Error, please set the POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DBNAME, and POSTGRES_VECS_COLLECTION environment variables to use pgvector database."
            )
        try:
            DB_CONNECTION = (
                f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
            )
            self.vx: Client = create_client(DB_CONNECTION)
        except Exception as e:
            raise ValueError(
                f"Error {e} occurred while attempting to connect to the pgvector provider with {DB_CONNECTION}."
            )
        self.vector_db_dimension = dimension
        self.collection_name = collection_name
        self.config: DatabaseConfig = config
        self.crypto_provider = crypto_provider
        super().__init__(config)

    def _initialize_vector_db(self) -> VectorDatabaseProvider:
        return PostgresVectorDBProvider(
            self.config,
            vx=self.vx,
            collection_name=self.collection_name,
            dimension=self.vector_db_dimension,
        )

    def _initialize_relational_db(self) -> RelationalDatabaseProvider:
        return PostgresRelationalDBProvider(
            self.config,
            vx=self.vx,
            crypto_provider=self.crypto_provider,
            collection_name=self.collection_name,
        )

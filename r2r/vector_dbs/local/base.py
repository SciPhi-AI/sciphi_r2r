import logging
import sqlite3
from typing import Optional, Union

from r2r.core import VectorDBProvider, VectorEntry, VectorSearchResult
import json

logger = logging.getLogger(__name__)


class LocalVectorDB(VectorDBProvider):
    def __init__(self, provider: str = "local", db_path='local_vector_db.sqlite') -> None:
        logger.info(
            "Initializing `LocalVectorDB` to store and retrieve embeddings."
        )

        super().__init__(provider)
        if provider != "local":
            raise ValueError(
                "LocalVectorDB must be initialized with provider `local`."
            )

        self.db_path = db_path
        self.collection_name: Optional[str] = None

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        return conn

    def _get_cursor(self, conn):
        return conn.cursor()

    def initialize_collection(
        self, collection_name: str, dimension: int
    ) -> None:
        conn = self._get_conn()
        cursor = self._get_cursor(conn)
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS "{collection_name}" (
                id TEXT PRIMARY KEY,
                vector TEXT,
                metadata TEXT
            )
        ''')
        self.collection_name = collection_name
        conn.commit()
        conn.close()

    def create_index(self, index_type, column_name, index_options):
        raise NotImplementedError(
            "LocalVectorDB does not support creating indexes."
        )

    def copy(self, entry: VectorEntry, commit=True) -> None:
            if self.collection_name is None:
                raise ValueError("Collection name is not set. Please call `initialize_collection` first.")
            
            conn = self._get_conn()
            cursor = self._get_cursor(conn)
            cursor.execute(f'''
                INSERT OR IGNORE INTO "{self.collection_name}" (id, vector, metadata)
                VALUES (?, ?, ?)
            ''', (str(entry.id), json.dumps(entry.vector), json.dumps(entry.metadata)))
            if commit:
                conn.commit()
            conn.close()

    def upsert(self, entry: VectorEntry, commit=True) -> None:
        if self.collection_name is None:
            raise ValueError("Collection name is not set. Please call `initialize_collection` first.")
        
        conn = self._get_conn()
        cursor = self._get_cursor(conn)
        cursor.execute(f'''
            INSERT OR REPLACE INTO "{self.collection_name}" (id, vector, metadata)
            VALUES (?, ?, ?)
        ''', (str(entry.id), json.dumps(entry.vector), json.dumps(entry.metadata)))
        if commit:
            conn.commit()
        conn.close()

    def search(
        self,
        query_vector: list[float],
        filters: dict[str, Union[bool, int, str]] = {},
        limit: int = 10,
        *args,
        **kwargs,
    ) -> list[VectorSearchResult]:
        
        if self.collection_name is None:
            raise ValueError("Collection name is not set. Please call `initialize_collection` first.")
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

        except ImportError:
            raise ImportError(
                "Please install numpy and scikit-learn to use the `search` method."
            )
        
        conn = self._get_conn()
        cursor = self._get_cursor(conn)
        cursor.execute(f'SELECT * FROM "{self.collection_name}"')
        results = []
        for id, vector, metadata in cursor.fetchall():
            print(f"parsing id = {id}")
            vector = json.loads(vector)
            metadata = json.loads(metadata)
            if all(metadata.get(k) == v for k, v in filters.items()):
                score = cosine_similarity(
                    np.array(query_vector).reshape(1, -1),
                    np.array(vector).reshape(1, -1)
                )[0][0]
                results.append(VectorSearchResult(id, score, metadata))
        results.sort(key=lambda x: x.score, reverse=True)
        conn.close()
        return results[:limit]

    def filtered_deletion(
        self, key: str, value: Union[bool, int, str]
    ) -> None:
        if self.collection_name is None:
            raise ValueError("Collection name is not set. Please call `initialize_collection` first.")
        
        conn = self._get_conn()
        cursor = self._get_cursor(conn)
        cursor.execute(f'SELECT * FROM "{self.collection_name}"')
        for id, vector, metadata in cursor.fetchall():
            metadata = json.loads(metadata)
            if metadata.get(key) == value:
                cursor.execute(f'DELETE FROM "{self.collection_name}" WHERE id = ?', (id,))
        conn.commit()
        conn.close()

    def close(self):
        pass        
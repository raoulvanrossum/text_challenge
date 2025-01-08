import uuid
import json
import os

from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm

from qdrant_client.http import models
from qdrant_client import QdrantClient


@dataclass
class ProcessedText:
    text: str
    embedding: List[float]
    language: str
    metadata: Optional[Dict] = None


@dataclass
class SearchResult:
    text: str
    score: float
    language: str
    metadata: Optional[Dict] = None


class TextIndexer:
    def __init__(
        self,
        collection_name: str = "patent_abstracts",
        dimension: int = 384,
        url: Optional[str] = None,
        load_from: Optional[str] = None,
    ):
        """Initialize the TextIndexer with Qdrant vector database

        Args:
            collection_name: Name of the Qdrant collection
            dimension: Dimension of the vectors (default 384 for multilingual-e5-small)
            url: Optional URL for Qdrant server (uses in-memory if None)
            load_from: Optional path to load a saved index
        """
        self.client = QdrantClient(url=url) if url else QdrantClient(":memory:")
        self.collection_name = collection_name
        self.dimension = dimension
        self.metadata = {}

        if load_from:
            self._load_state(load_from)
        else:
            self._ensure_collection()

    @property
    def ntotal(self) -> int:
        """Get total number of vectors in the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return 0

    def _ensure_collection(self):
        """Ensure collection exists with proper settings"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.dimension, distance=models.Distance.COSINE
                    ),
                )
                logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise

    def add_texts(self, processed_texts: List[ProcessedText]) -> None:
        """Add processed texts to the index

        Args:
            processed_texts: List of ProcessedText objects containing text and embeddings
        """
        try:
            points = []
            # Add progress bar for processing texts
            for pt in tqdm(
                processed_texts, desc="Processing texts for indexing", unit="text"
            ):
                point_id = str(uuid.uuid4())
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=pt.embedding,
                        payload={
                            "text": pt.text,
                            "language": pt.language,
                            "metadata": pt.metadata or {},
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                )
                self.metadata[point_id] = pt.metadata or {}

            logger.info(f"Inserting {len(points)} points into the index...")
            # Use tqdm to show progress during upsert
            batch_size = 100
            for i in tqdm(
                range(0, len(points), batch_size),
                desc="Uploading to Qdrant",
                unit="batch",
            ):
                batch = points[i : i + batch_size]
                self.client.upsert(collection_name=self.collection_name, points=batch)

            logger.info(f"Successfully added {len(points)} texts to the index")
        except Exception as e:
            logger.error(f"Error adding texts: {e}")
            raise

    def search(
        self, query_embedding: List[float], top_k: int = 5, threshold: float = 0.7
    ) -> List[SearchResult]:
        """Search for similar texts

        Args:
            query_embedding: Vector representation of the query
            top_k: Number of results to return
            threshold: Minimum similarity score threshold

        Returns:
            List of SearchResult objects
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=threshold,
            )

            results.sort(key=lambda x: x.score, reverse=True)

            return [
                SearchResult(
                    text=r.payload["text"],
                    score=r.score,
                    language=r.payload["language"],
                    metadata=r.payload.get("metadata", {}),
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def save_index(self, path: str) -> None:
        """Save index state to disk

        Args:
            path: Directory path to save the index
        """
        try:
            os.makedirs(path, exist_ok=True)

            # Save metadata and configuration
            state = {
                "metadata": self.metadata,
                "config": {
                    "collection_name": self.collection_name,
                    "dimension": self.dimension,
                },
            }

            with open(os.path.join(path, "state.json"), "w") as f:
                json.dump(state, f)

            # Export vectors and their payload
            vectors_data = []
            offset = None

            with tqdm(desc="Exporting vectors", unit="batch") as pbar:
                while True:
                    batch, offset = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=100,
                        offset=offset,
                        with_vectors=True,
                        with_payload=True,
                    )

                    if not batch:
                        break

                    for point in batch:
                        if point.vector is None:
                            logger.warning(f"Skipping point {point.id} with no vector")
                            continue

                        vector_list = (
                            point.vector.tolist()
                            if hasattr(point.vector, "tolist")
                            else list(point.vector)
                        )
                        vectors_data.append(
                            {
                                "id": point.id,
                                "vector": vector_list,
                                "payload": point.payload,
                            }
                        )

                    pbar.update(len(batch))

                    if offset is None:
                        break

            with open(os.path.join(path, "vectors.json"), "w") as f:
                json.dump(vectors_data, f)

            logger.info(f"Saved index to {path} with {len(vectors_data)} vectors")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise

    def _load_state(self, path: str) -> None:
        """Load index state from disk

        Args:
            path: Directory path to load the index from
        """
        try:
            # Load metadata and config
            with open(os.path.join(path, "state.json"), "r") as f:
                state = json.load(f)

            self.metadata = state["metadata"]
            self.collection_name = state["config"]["collection_name"]
            self.dimension = state["config"]["dimension"]

            self._ensure_collection()

            # Load vectors
            with open(os.path.join(path, "vectors.json"), "r") as f:
                vectors_data = json.load(f)

            # Insert vectors in batches with progress bar
            batch_size = 100
            total_batches = (len(vectors_data) + batch_size - 1) // batch_size

            for i in tqdm(
                range(0, len(vectors_data), batch_size),
                desc="Loading vectors",
                total=total_batches,
                unit="batch",
            ):
                batch = vectors_data[i : i + batch_size]
                points = []
                for item in batch:
                    if not isinstance(item["vector"], list):
                        logger.warning(
                            f"Skipping invalid vector format for item {item['id']}"
                        )
                        continue

                    if len(item["vector"]) != self.dimension:
                        logger.warning(
                            f"Skipping vector with wrong dimension for item {item['id']}"
                        )
                        continue

                    points.append(
                        models.PointStruct(
                            id=item["id"],
                            vector=item["vector"],
                            payload=item["payload"],
                        )
                    )

                if points:
                    self.client.upsert(
                        collection_name=self.collection_name, points=points
                    )

            logger.info(f"Loaded index from {path} with {len(vectors_data)} vectors")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise

    def get_languages_summary(self) -> Dict[str, int]:
        """
        Get a summary of languages in the index.

        Returns:
            Dict[str, int]: Dictionary with language codes as keys and counts as values
        """
        try:
            language_counts = {}
            offset = None

            while True:
                batch, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                )

                if not batch:
                    break

                for point in batch:
                    language = point.payload.get("language", "unknown")
                    language_counts[language] = language_counts.get(language, 0) + 1

                if offset is None:
                    break

            return language_counts

        except Exception as e:
            logger.error(f"Error getting language summary: {e}")
            return {}

    def __len__(self) -> int:
        """Return the number of documents in the index"""
        return self.ntotal

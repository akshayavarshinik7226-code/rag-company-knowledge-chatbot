"""Persistent vector-store implementations used by the RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent


def distance_to_similarity(distance: float | None) -> float | None:
    """Convert a non-negative vector distance into a comparable 0..1 score.

    Both configured stores use squared-L2 distance for the normalized sentence
    embeddings.  ``1 / (1 + distance)`` is a monotonic, bounded score: 1 means
    an exact vector match and larger distances produce lower scores.
    """
    if distance is None:
        return None
    return round(1.0 / (1.0 + float(distance)), 4)


class VectorStore(ABC):
    @abstractmethod
    def add(self, documents: list[dict[str, Any]], embeddings: list[list[float]]) -> int:
        """Persist chunks and their embeddings."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        """Return the nearest stored chunks with a documented similarity score."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of persisted chunks."""


class ChromaVectorStore(VectorStore):
    def __init__(self) -> None:
        db_dir = BASE_DIR / "chroma_db"
        self.collection = chromadb.PersistentClient(path=str(db_dir)).get_or_create_collection(
            name="company_knowledge"
        )

    def add(self, documents: list[dict[str, Any]], embeddings: list[list[float]]) -> int:
        ids = [document["id"] for document in documents]
        texts = [document["text"] for document in documents]
        metadatas = [
            {key: str(value) for key, value in document["metadata"].items() if value is not None}
            for document in documents
        ]
        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        return len(documents)

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        if not self.count():
            return []
        result = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            {
                "text": text,
                "source": (metadatas[index] or {}).get("source", "Unknown"),
                "page": (metadatas[index] or {}).get("page"),
                "similarity_score": distance_to_similarity(
                    distances[index] if index < len(distances) else None
                ),
            }
            for index, text in enumerate(documents)
        ]

    def count(self) -> int:
        return self.collection.count()


class FaissVectorStore(VectorStore):
    """A small persistent FAISS store suitable for this local project."""

    def __init__(self) -> None:
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError(
                "VECTOR_DB=faiss requires faiss-cpu. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from error

        self.faiss = faiss
        self.db_dir = BASE_DIR / "faiss_db"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.db_dir / "company_knowledge.index"
        self.metadata_path = self.db_dir / "company_knowledge.json"
        self.dimension = 384
        self.index = self._load_index()
        self.records = self._load_records()

    def _load_index(self):
        if self.index_path.exists():
            return self.faiss.read_index(str(self.index_path))
        return self.faiss.IndexIDMap2(self.faiss.IndexFlatL2(self.dimension))

    def _load_records(self) -> dict[str, dict[str, Any]]:
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    @staticmethod
    def _numeric_id(document_id: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(document_id.encode("utf-8"), digest_size=8).digest(), "big"
        ) & ((1 << 63) - 1)

    def _save(self) -> None:
        temp_index = self.index_path.with_suffix(".tmp")
        temp_metadata = self.metadata_path.with_suffix(".tmp")
        self.faiss.write_index(self.index, str(temp_index))
        temp_metadata.write_text(json.dumps(self.records, ensure_ascii=False), encoding="utf-8")
        os.replace(temp_index, self.index_path)
        os.replace(temp_metadata, self.metadata_path)

    def add(self, documents: list[dict[str, Any]], embeddings: list[list[float]]) -> int:
        latest = {document["id"]: (document, embedding) for document, embedding in zip(documents, embeddings)}
        numeric_ids = np.array([self._numeric_id(document_id) for document_id in latest], dtype="int64")
        if len(numeric_ids):
            self.index.remove_ids(numeric_ids)
            vectors = np.asarray([item[1] for item in latest.values()], dtype="float32")
            self.index.add_with_ids(vectors, numeric_ids)
        for document_id, (document, _) in latest.items():
            self.records[str(self._numeric_id(document_id))] = {
                "text": document["text"],
                **document["metadata"],
            }
        self._save()
        return len(latest)

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        if not self.count():
            return []
        distances, ids = self.index.search(np.asarray([query_embedding], dtype="float32"), top_k)
        results = []
        for distance, numeric_id in zip(distances[0], ids[0]):
            record = self.records.get(str(int(numeric_id)))
            if record is None or numeric_id == -1:
                continue
            results.append(
                {
                    "text": record["text"],
                    "source": record.get("source", "Unknown"),
                    "page": record.get("page"),
                    "similarity_score": distance_to_similarity(float(distance)),
                }
            )
        return results

    def count(self) -> int:
        return self.index.ntotal


def get_vector_store(provider_name: str | None = None) -> VectorStore:
    load_dotenv(BASE_DIR / ".env")
    provider = (provider_name or os.getenv("VECTOR_DB", "chroma")).strip().lower()
    if provider == "chroma":
        return ChromaVectorStore()
    if provider == "faiss":
        return FaissVectorStore()
    raise RuntimeError("Unsupported VECTOR_DB. Use `chroma` or `faiss`.")

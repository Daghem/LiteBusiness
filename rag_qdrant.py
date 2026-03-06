import os
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List

import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    regime: str
    source: str
    chunk_id: int
    text: str
    score: float


@dataclass(frozen=True)
class CorpusConfig:
    regime_id: str
    label: str
    path: Path


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text], batch_size=1)[0]


class QdrantRAG:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str | None,
        collection_name: str,
        embedding_model: str,
    ) -> None:
        self.collection_name = collection_name
        self.embedder = SentenceTransformerEmbedder(embedding_model)
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)

    @classmethod
    def from_env(cls) -> "QdrantRAG":
        return cls(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION", "flytax_normativa_2026"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
        )

    @staticmethod
    def normalize_regime_id(value: str) -> str:
        text = value.strip().lower()
        text = re.sub(r"^normativo_", "", text)
        text = re.sub(r"_agg_\d{4}$", "", text)
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if text == "forfettari":
            return "forfettario"
        return text or "generico"

    @classmethod
    def derive_corpus_config(cls, pdf_dir: Path) -> CorpusConfig:
        pdf_dir = Path(pdf_dir)
        regime_id = cls.normalize_regime_id(pdf_dir.name)
        label = regime_id.replace("_", " ").title()
        return CorpusConfig(regime_id=regime_id, label=label, path=pdf_dir)

    @classmethod
    def discover_pdf_corpora(cls, base_dir: Path = Path(".")) -> List[CorpusConfig]:
        corpora = []
        for candidate in sorted(Path(base_dir).glob("Normativo_*")):
            if not candidate.is_dir():
                continue
            if any(candidate.glob("*.pdf")):
                corpora.append(cls.derive_corpus_config(candidate))
        return corpora

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if overlap >= chunk_size:
            raise ValueError("overlap deve essere minore di chunk_size")

        chunks: List[str] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            if end < text_len:
                last_space = chunk.rfind(" ")
                if last_space > chunk_size * 0.6:
                    end = start + last_space
                    chunk = text[start:end]
            chunks.append(chunk.strip())
            if end >= text_len:
                break
            start = max(0, end - overlap)
        return [c for c in chunks if c]

    @staticmethod
    def extract_text_from_pdf(pdf_path: Path) -> str:
        pages: List[str] = []
        with fitz.open(pdf_path) as document:
            for page in document:
                pages.append(page.get_text())
        return "".join(pages)

    def _collection_exists(self) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        exists = self._collection_exists()
        if recreate and exists:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
                on_disk_payload=True,
            )

    def build_from_pdf_directory(
        self,
        pdf_dir: Path = Path("Normativo_Forfettari_Agg_2026"),
        regime_id: str | None = None,
        chunk_size: int = 1200,
        overlap: int = 200,
        embed_batch_size: int = 32,
        recreate_collection: bool = True,
    ) -> int:
        corpus = self.derive_corpus_config(Path(pdf_dir))
        if regime_id:
            corpus = CorpusConfig(regime_id=self.normalize_regime_id(regime_id), label=corpus.label, path=corpus.path)
        return self.build_from_pdf_directories(
            corpora=[corpus],
            chunk_size=chunk_size,
            overlap=overlap,
            embed_batch_size=embed_batch_size,
            recreate_collection=recreate_collection,
        )

    def build_from_pdf_directories(
        self,
        corpora: Iterable[CorpusConfig],
        chunk_size: int = 1200,
        overlap: int = 200,
        embed_batch_size: int = 32,
        recreate_collection: bool = True,
    ) -> int:
        raw_chunks = []
        for corpus in corpora:
            pdf_files = sorted(corpus.path.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"Nessun PDF trovato in {corpus.path}")
            for pdf_path in pdf_files:
                text = self.extract_text_from_pdf(pdf_path)
                chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                for chunk_id, chunk_text in enumerate(chunks):
                    raw_chunks.append(
                        {
                            "regime": corpus.regime_id,
                            "source": pdf_path.name,
                            "chunk_id": chunk_id,
                            "text": chunk_text,
                        }
                    )

        if not raw_chunks:
            raise ValueError("Nessun chunk generato dai PDF")

        probe_vec = self.embedder.embed_query(raw_chunks[0]["text"])
        self.ensure_collection(vector_size=len(probe_vec), recreate=recreate_collection)

        point_id = 1
        for start in range(0, len(raw_chunks), embed_batch_size):
            batch = raw_chunks[start : start + embed_batch_size]
            vectors = self.embedder.embed_texts(
                [item["text"] for item in batch],
                batch_size=embed_batch_size,
            )
            points = []
            for item, vector in zip(batch, vectors):
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "regime": item["regime"],
                            "source": item["source"],
                            "chunk_id": item["chunk_id"],
                            "text": item["text"],
                        },
                    )
                )
                point_id += 1

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        return len(raw_chunks)

    def load(self) -> None:
        if not self._collection_exists():
            raise FileNotFoundError(
                f"Collection Qdrant non trovata: {self.collection_name}"
            )
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            raise ValueError(f"Collection Qdrant vuota: {self.collection_name}")

    def search(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 0.2,
        regime_ids: List[str] | None = None,
    ) -> List[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []

        query_vector = self.embedder.embed_query(query)
        query_filter = None
        if regime_ids:
            normalized_regimes = [self.normalize_regime_id(item) for item in regime_ids if item]
            if normalized_regimes:
                if len(normalized_regimes) == 1:
                    match = models.MatchValue(value=normalized_regimes[0])
                else:
                    match = models.MatchAny(any=normalized_regimes)
                query_filter = models.Filter(
                    must=[models.FieldCondition(key="regime", match=match)]
                )

        # Compatibilita' tra versioni del client:
        # - nuove: query_points(...)
        # - vecchie: search(...)
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                score_threshold=min_score,
                query_filter=query_filter,
            )
            hits = response.points
        else:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                score_threshold=min_score,
                query_filter=query_filter,
            )

        results: List[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            regime = payload.get("regime")
            text = payload.get("text")
            source = payload.get("source")
            chunk_id = payload.get("chunk_id")
            if regime is None or text is None or source is None or chunk_id is None:
                continue
            results.append(
                RetrievedChunk(
                    regime=str(regime),
                    source=str(source),
                    chunk_id=int(chunk_id),
                    text=str(text),
                    score=float(hit.score),
                )
            )
        return results

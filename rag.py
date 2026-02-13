import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass
class RetrievedChunk:
    source: str
    chunk_id: int
    text: str
    score: float


class LocalRAG:
    def __init__(self, index_file: Path = Path("rag_index/index.json")) -> None:
        self.index_file = Path(index_file)
        self.idf: Dict[str, float] = {}
        self.chunks: List[Dict] = []

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
    def tokenize(text: str) -> List[str]:
        return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 1]

    @staticmethod
    def tf(tokens: List[str]) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        total = len(tokens)
        if total == 0:
            return {}
        return {term: count / total for term, count in counts.items()}

    @staticmethod
    def normalize(vector: Dict[str, float]) -> Dict[str, float]:
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return vector
        return {term: value / norm for term, value in vector.items()}

    def build_from_directory(
        self,
        text_dir: Path = Path("testi_estratti_2026"),
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> None:
        text_dir = Path(text_dir)
        files = sorted(text_dir.glob("*.txt"))
        files = [f for f in files if f.name != "tutti_i_documenti.txt"]
        if not files:
            raise FileNotFoundError(f"Nessun .txt trovato in {text_dir}")

        raw_chunks: List[Dict] = []
        for file_path in files:
            content = file_path.read_text(encoding="utf-8")
            split_chunks = self.chunk_text(content, chunk_size=chunk_size, overlap=overlap)
            for chunk_id, chunk_text in enumerate(split_chunks):
                raw_chunks.append(
                    {
                        "source": file_path.name,
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                    }
                )

        num_docs = len(raw_chunks)
        doc_freq: Dict[str, int] = {}
        chunk_tf: List[Dict[str, float]] = []

        for chunk in raw_chunks:
            tokens = self.tokenize(chunk["text"])
            tf_values = self.tf(tokens)
            chunk_tf.append(tf_values)
            for term in tf_values:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        self.idf = {
            term: math.log((num_docs + 1) / (df + 1)) + 1
            for term, df in doc_freq.items()
        }

        self.chunks = []
        for chunk, tf_values in zip(raw_chunks, chunk_tf):
            weighted = {
                term: tf_value * self.idf.get(term, 0.0)
                for term, tf_value in tf_values.items()
            }
            normalized = self.normalize(weighted)
            self.chunks.append(
                {
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "vector": normalized,
                }
            )

    def save(self) -> None:
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"idf": self.idf, "chunks": self.chunks}
        self.index_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        if not self.index_file.exists():
            raise FileNotFoundError(f"Indice non trovato: {self.index_file}")
        payload = json.loads(self.index_file.read_text(encoding="utf-8"))
        self.idf = payload.get("idf", {})
        self.chunks = payload.get("chunks", [])

    def search(self, query: str, top_k: int = 4, min_score: float = 0.08) -> List[RetrievedChunk]:
        query_tokens = self.tokenize(query)
        query_tf = self.tf(query_tokens)
        query_weighted = {
            term: tf_value * self.idf.get(term, 0.0)
            for term, tf_value in query_tf.items()
            if term in self.idf
        }
        query_vec = self.normalize(query_weighted)
        if not query_vec:
            return []

        results: List[RetrievedChunk] = []
        for chunk in self.chunks:
            chunk_vec = chunk["vector"]
            if len(query_vec) <= len(chunk_vec):
                score = sum(value * chunk_vec.get(term, 0.0) for term, value in query_vec.items())
            else:
                score = sum(query_vec.get(term, 0.0) * value for term, value in chunk_vec.items())
            if score >= min_score:
                results.append(
                    RetrievedChunk(
                        source=chunk["source"],
                        chunk_id=chunk["chunk_id"],
                        text=chunk["text"],
                        score=score,
                    )
                )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class LexicalChunk:
    regime: str
    source: str
    chunk_id: int
    text: str
    page_start: int | None = None
    page_end: int | None = None


class LexicalFallbackIndex:
    def __init__(self, chunks: List[LexicalChunk]) -> None:
        self.chunks = chunks
        self._token_cache = [set(self.tokenize(chunk.text)) for chunk in chunks]

    @classmethod
    def from_chunks(cls, chunks: Iterable[LexicalChunk]) -> "LexicalFallbackIndex":
        return cls(list(chunks))

    @classmethod
    def from_local_index(cls, index_path: Path) -> "LexicalFallbackIndex | None":
        if not index_path.exists():
            return None
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        raw_chunks = payload.get("chunks", [])
        chunks: List[LexicalChunk] = []
        for item in raw_chunks:
            source = str(item.get("source", ""))
            if source.endswith(".txt"):
                source = source[:-4] + ".pdf"
            chunks.append(
                LexicalChunk(
                    regime="forfettario",
                    source=source,
                    chunk_id=int(item.get("chunk_id", 0)),
                    text=str(item.get("text", "")),
                    page_start=None,
                    page_end=None,
                )
            )
        return cls(chunks) if chunks else None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 1]

    def search(
        self,
        query: str,
        top_k: int = 6,
        regime_id: str | None = None,
    ) -> List[tuple[LexicalChunk, float]]:
        tokens = self.tokenize(query)
        if not tokens:
            return []
        query_set = set(tokens)

        scored: List[tuple[LexicalChunk, float]] = []
        for chunk, token_set in zip(self.chunks, self._token_cache):
            if regime_id and chunk.regime != regime_id:
                continue
            overlap = len(query_set.intersection(token_set))
            if overlap == 0:
                continue
            score = overlap / math.sqrt(len(query_set) * max(len(token_set), 1))
            scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def find_mentions(
        self,
        term: str,
        regime_id: str | None = None,
        max_hits: int = 50,
    ) -> List[LexicalChunk]:
        if not term:
            return []
        parts = [part for part in re.split(r"\s+", term.strip()) if part]
        if not parts:
            return []
        escaped = r"\s+".join(re.escape(part) for part in parts)
        pattern = re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)
        hits: List[LexicalChunk] = []
        for chunk in self.chunks:
            if regime_id and chunk.regime != regime_id:
                continue
            if pattern.search(chunk.text):
                hits.append(chunk)
                if len(hits) >= max_hits:
                    break
        return hits

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app_models import AdminStats, ChatMessage, ChatSummary, ChatTranscript


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatHistoryStore:
    def __init__(self, base_dir: Path = Path("data/chat_history")) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_chat_id(self) -> str:
        return uuid4().hex

    def _chat_path(self, chat_id: str) -> Path:
        return self.base_dir / f"{chat_id}.json"

    def save_turn(
        self,
        chat_id: str,
        regime_id: str | None,
        user_message: str,
        assistant_message: str,
        assistant_sources: list[str],
        confidence_label: str | None = None,
        confidence_score: float | None = None,
        retrieval_mode: str | None = None,
    ) -> dict:
        chat = self._load_raw(chat_id) or {
            "chat_id": chat_id,
            "title": user_message.strip()[:72] or "Nuova chat",
            "regime_id": regime_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "messages": [],
        }
        timestamp = utc_now_iso()
        chat["regime_id"] = regime_id or chat.get("regime_id")
        chat["updated_at"] = timestamp
        chat.setdefault("messages", []).extend(
            [
                {
                    "role": "user",
                    "text": user_message,
                    "sources": [],
                    "created_at": timestamp,
                },
                {
                    "role": "assistant",
                    "text": assistant_message,
                    "sources": assistant_sources,
                    "created_at": timestamp,
                    "confidence_label": confidence_label,
                    "confidence_score": confidence_score,
                    "retrieval_mode": retrieval_mode,
                },
            ]
        )
        self._chat_path(chat_id).write_text(
            json.dumps(chat, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return chat

    def list_chats(self, limit: int = 30) -> list[ChatSummary]:
        chats = []
        for path in self.base_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            chats.append(
                ChatSummary(
                    chat_id=payload["chat_id"],
                    title=payload.get("title", "Chat"),
                    regime_id=payload.get("regime_id"),
                    updated_at=payload.get("updated_at", ""),
                    message_count=len(payload.get("messages", [])),
                )
            )
        chats.sort(key=lambda item: item.updated_at, reverse=True)
        return chats[:limit]

    def get_chat(self, chat_id: str) -> ChatTranscript | None:
        payload = self._load_raw(chat_id)
        if payload is None:
            return None
        return ChatTranscript(
            chat_id=payload["chat_id"],
            title=payload.get("title", "Chat"),
            regime_id=payload.get("regime_id"),
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
            messages=[ChatMessage(**message) for message in payload.get("messages", [])],
        )

    def delete_chat(self, chat_id: str) -> bool:
        path = self._chat_path(chat_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def _load_raw(self, chat_id: str) -> dict | None:
        path = self._chat_path(chat_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


class JsonlStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: dict) -> None:
        record = {"timestamp": utc_now_iso(), **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


class FeedbackStore(JsonlStore):
    pass


class EventStore(JsonlStore):
    pass


def build_admin_stats(
    chat_store: ChatHistoryStore,
    feedback_store: FeedbackStore,
    event_store: EventStore,
) -> AdminStats:
    chats = chat_store.list_chats(limit=500)
    total_messages = 0
    question_counter: Counter[str] = Counter()
    regime_counter: Counter[str] = Counter()

    for summary in chats:
        transcript = chat_store.get_chat(summary.chat_id)
        if transcript is None:
            continue
        total_messages += len(transcript.messages)
        if transcript.regime_id:
            regime_counter[transcript.regime_id] += 1
        for message in transcript.messages:
            if message.role == "user":
                question_counter[message.text.strip()] += 1

    feedback_records = feedback_store.read_all()
    positive_feedback = sum(1 for item in feedback_records if item.get("vote") == "up")
    negative_feedback = sum(1 for item in feedback_records if item.get("vote") == "down")

    event_records = event_store.read_all()
    no_result_events = sum(1 for item in event_records if item.get("event") == "rag_no_results")
    low_confidence_events = sum(
        1 for item in event_records if item.get("event") == "rag_low_confidence"
    )

    return AdminStats(
        total_chats=len(chats),
        total_messages=total_messages,
        positive_feedback=positive_feedback,
        negative_feedback=negative_feedback,
        no_result_events=no_result_events,
        low_confidence_events=low_confidence_events,
        top_questions=[item for item, _ in question_counter.most_common(5)],
        top_regimes=[item for item, _ in regime_counter.most_common(5)],
    )

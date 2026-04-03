import tempfile
import unittest
from pathlib import Path

from storage_services import ChatHistoryStore, EventStore, FeedbackStore, build_admin_stats


class StorageServicesTests(unittest.TestCase):
    def test_chat_history_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChatHistoryStore(Path(tmpdir) / "history")
            store.save_turn(
                chat_id="chat-1",
                regime_id="forfettario",
                user_message="Quanto posso fatturare?",
                assistant_message="La soglia ordinaria è 85.000 euro.",
                assistant_sources=["02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf"],
            )
            transcript = store.get_chat("chat-1")
            self.assertIsNotNone(transcript)
            self.assertEqual(len(transcript.messages), 2)
            self.assertEqual(transcript.regime_id, "forfettario")

    def test_admin_stats_aggregate_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            chat_store = ChatHistoryStore(base / "history")
            feedback_store = FeedbackStore(base / "feedback" / "feedback.jsonl")
            event_store = EventStore(base / "events" / "events.jsonl")

            chat_store.save_turn(
                chat_id="chat-1",
                regime_id="forfettario",
                user_message="Domanda 1",
                assistant_message="Risposta 1",
                assistant_sources=[],
            )
            feedback_store.append({"vote": "up", "message": "Risposta 1"})
            event_store.append({"event": "rag_no_results"})

            stats = build_admin_stats(chat_store, feedback_store, event_store)
            self.assertEqual(stats.total_chats, 1)
            self.assertEqual(stats.positive_feedback, 1)
            self.assertEqual(stats.no_result_events, 1)


if __name__ == "__main__":
    unittest.main()

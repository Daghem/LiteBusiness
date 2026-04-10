import asyncio
import importlib
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

class FakeRag:
    def __init__(self, search_results=None, payload_chunks=None):
        self._search_results = search_results
        self._payload_chunks = payload_chunks or []

    def load(self):
        return None

    def search(self, query, top_k=4, min_score=0.2, regime_ids=None):
        if self._search_results is not None:
            return self._search_results
        regime = regime_ids[0] if regime_ids else "forfettario"
        return [
            types.SimpleNamespace(
                regime=regime,
                source="04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                chunk_id=0,
                text="Test chunk",
                score=0.9,
                page_start=None,
                page_end=None,
            )
        ]

    def iter_payload_chunks(self, regime_ids=None, batch_size=256):
        return iter(self._payload_chunks)


class ApiDeepseekRoutingTests(unittest.TestCase):
    def load_module(
        self,
        llm_answer="Il CONTEXT fornito non contiene informazioni specifiche.",
        corpora=None,
        rag_results=None,
        payload_chunks=None,
        extra_env=None,
    ):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=llm_answer)
                )
            ]
        )

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = mock.Mock(return_value=fake_client)
        fake_openai.APIError = type("APIError", (Exception,), {})
        fake_openai.RateLimitError = type("RateLimitError", (Exception,), {})

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        fake_pydantic = types.ModuleType("pydantic")

        class FakeBaseModel:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def model_dump(self):
                return self.__dict__.copy()

        fake_pydantic.BaseModel = FakeBaseModel
        fake_pydantic.Field = lambda default=None, default_factory=None: (
            default_factory() if default_factory is not None else default
        )

        fake_fastapi = types.ModuleType("fastapi")

        class FakeFastAPI:
            def add_middleware(self, *args, **kwargs):
                return None

            def post(self, *args, **kwargs):
                def decorator(fn):
                    return fn

                return decorator

            def get(self, *args, **kwargs):
                def decorator(fn):
                    return fn

                return decorator

            def delete(self, *args, **kwargs):
                def decorator(fn):
                    return fn

                return decorator

        fake_fastapi.FastAPI = FakeFastAPI
        fake_fastapi.File = lambda *args, **kwargs: None
        fake_fastapi.Header = lambda *args, **kwargs: None

        class FakeHTTPException(Exception):
            def __init__(self, status_code=400, detail=""):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class FakeUploadFile:
            def __init__(self, filename="test.pdf"):
                self.filename = filename

            async def read(self):
                return b""

        fake_fastapi.HTTPException = FakeHTTPException
        fake_fastapi.UploadFile = FakeUploadFile

        fake_fastapi_middleware = types.ModuleType("fastapi.middleware")
        fake_fastapi_cors = types.ModuleType("fastapi.middleware.cors")
        fake_fastapi_cors.CORSMiddleware = object
        fake_fastapi_responses = types.ModuleType("fastapi.responses")
        fake_fastapi_responses.FileResponse = lambda *args, **kwargs: {
            "path": str(args[0]) if args else "",
            "kwargs": kwargs,
        }

        fake_app_models = types.ModuleType("app_models")
        for class_name in (
            "ChatRequest",
            "ChatResponse",
            "ChatSummary",
            "ChatTranscript",
            "ChatTurnPayload",
            "FeedbackRequest",
            "FeedbackResponse",
            "RegimeOption",
            "SimulationRequest",
            "SimulationResponse",
            "SourceRef",
        ):
            fake_class = type(class_name, (FakeBaseModel,), {})
            setattr(fake_app_models, class_name, fake_class)

        fake_storage_services = types.ModuleType("storage_services")

        class FakeStore:
            def __init__(self, *args, **kwargs):
                self.items = []

            def append(self, payload):
                self.items.append(payload)

            def read_all(self):
                return list(self.items)

            def list_chats(self, limit=30):
                return []

            def get_chat(self, chat_id):
                return None

            def delete_chat(self, chat_id):
                return True

            def save_turn(self, **kwargs):
                return {
                    "chat_id": kwargs["chat_id"],
                    "title": "Chat",
                    "regime_id": kwargs.get("regime_id"),
                    "created_at": "",
                    "updated_at": "",
                    "messages": [],
                }

        fake_storage_services.ChatHistoryStore = FakeStore
        fake_storage_services.EventStore = FakeStore
        fake_storage_services.FeedbackStore = FakeStore

        class FakeAdminStats(FakeBaseModel):
            top_questions = []
            top_regimes = []

        fake_storage_services.build_admin_stats = lambda *args, **kwargs: FakeAdminStats(
            total_chats=0,
            total_messages=0,
            positive_feedback=0,
            negative_feedback=0,
            no_result_events=0,
            low_confidence_events=0,
            top_questions=[],
            top_regimes=[],
        )

        fake_tax_simulator = types.ModuleType("tax_simulator")
        fake_tax_simulator.simulate_forfettario = lambda payload: payload

        fake_rag_qdrant = types.ModuleType("rag_qdrant")
        fake_rag_qdrant.CorpusConfig = types.SimpleNamespace
        fake_rag_qdrant.RetrievedChunk = types.SimpleNamespace

        class FakeQdrantRAG:
            @classmethod
            def from_env(cls):
                return FakeRag(
                    search_results=rag_results,
                    payload_chunks=payload_chunks,
                )

            @classmethod
            def discover_pdf_corpora(cls, base_dir=None):
                if corpora is not None:
                    return corpora
                return [
                    types.SimpleNamespace(
                        regime_id="forfettario",
                        label="Regime Forfettario",
                        path="Normativo_Forfettari_Agg_2026",
                    )
                ]

        fake_rag_qdrant.QdrantRAG = FakeQdrantRAG

        env_overrides = {"API_KEY_DEEPSEEK": "test-key", "LEXICAL_FALLBACK_ENABLED": "0"}
        if extra_env:
            env_overrides.update(extra_env)
        with mock.patch.dict(os.environ, env_overrides, clear=False):
            with mock.patch.dict(
                sys.modules,
                {
                    "dotenv": fake_dotenv,
                    "fastapi": fake_fastapi,
                    "fastapi.middleware": fake_fastapi_middleware,
                    "fastapi.middleware.cors": fake_fastapi_cors,
                    "fastapi.responses": fake_fastapi_responses,
                    "openai": fake_openai,
                    "pydantic": fake_pydantic,
                    "app_models": fake_app_models,
                    "rag_qdrant": fake_rag_qdrant,
                    "storage_services": fake_storage_services,
                    "tax_simulator": fake_tax_simulator,
                },
                clear=False,
            ):
                sys.modules.pop("api_deepseek", None)
                module = importlib.import_module("api_deepseek")
        return module

    def ask(self, module, question):
        return asyncio.run(
            module.read_root(
                module.ChatRequest(content=question, regime_id=None, chat_id=None)
            )
        )

    def test_loss_query_no_longer_hits_aliquota_5_branch(self):
        module = self.load_module()
        response = self.ask(module, "Quando si perde l'agevolazione del 35%?")
        self.assertIn("35%", response.message)
        self.assertNotIn("aliquota del 5%", response.message.lower())

    def test_cash_basis_query_does_not_hit_inps_branch(self):
        module = self.load_module()
        response = self.ask(module, "Conta il fatturato o l'incassato per le soglie?")
        self.assertIn("criterio di cassa", response.message.lower())
        self.assertNotIn("riduzione inps del 35%", response.message.lower())

    def test_reapply_query_returns_inps_answer(self):
        module = self.load_module()
        response = self.ask(module, "Se rinuncio poi posso richiederla di nuovo?")
        self.assertIn("nuova domanda", response.message.lower())
        self.assertNotIn("riformula la domanda", response.message.lower())

    def test_generic_deadline_follow_up_maps_to_inps_35(self):
        module = self.load_module()
        response = self.ask(module, "Entro quando va presentata la domanda?")
        self.assertIn("28 febbraio", response.message.lower())

    def test_generic_march_follow_up_maps_to_inps_35(self):
        module = self.load_module()
        response = self.ask(module, "Se la invio il 10 marzo quando decorre?")
        self.assertIn("1° gennaio dell'anno successivo", response.message.lower())

    def test_new_activity_query_mentions_iscrizione_previdenziale(self):
        module = self.load_module()
        response = self.ask(module, "Per nuove attivita quando va fatta la domanda 35%?")
        self.assertIn("iscrizione", response.message.lower())
        self.assertIn("gestione artigiani e commercianti", response.message.lower())

    def test_generic_new_activity_follow_up_maps_to_inps_35(self):
        module = self.load_module()
        response = self.ask(module, "Per nuove attivita quando va fatta la domanda?")
        self.assertIn("gestione artigiani e commercianti", response.message.lower())

    def test_renewal_query_returns_specific_answer(self):
        module = self.load_module()
        response = self.ask(module, "La riduzione si rinnova automaticamente?")
        self.assertIn("si rinnova", response.message.lower())
        self.assertNotIn("context", response.message.lower())

    def test_late_deadline_query_returns_specific_answer(self):
        module = self.load_module()
        response = self.ask(module, "Riduzione 35: posso fare domanda dopo il 28 febbraio?")
        self.assertIn("puoi presentarla", response.message.lower())
        self.assertIn("anno successivo", response.message.lower())

    def test_ex_datore_after_three_years_is_not_ostativa(self):
        module = self.load_module()
        response = self.ask(
            module,
            "Se fatturo al 55% a ex datore e rapporto cessato da 3 anni, e ostativo?",
        )
        self.assertTrue(response.message.lower().startswith("no"))
        self.assertIn("non è ostativo", response.message.lower())

    def test_srl_control_query_is_handled_in_domain(self):
        module = self.load_module()
        response = self.ask(
            module,
            "Se ho SRL al 30% ma controllo di fatto, posso restare?",
        )
        self.assertIn("controllo", response.message.lower())
        self.assertNotIn("riformula la domanda", response.message.lower())

    def test_llm_answer_is_sanitized(self):
        module = self.load_module(
            llm_answer="Il CONTEXT fornito non contiene informazioni specifiche sui regimi speciali IVA."
        )
        response = self.ask(module, "Regimi speciali IVA incompatibili: quali?")
        self.assertNotIn("CONTEXT", response.message)
        self.assertIn("documenti disponibili", response.message.lower())

    def test_explicit_other_regime_uses_generic_rag_path(self):
        module = self.load_module(
            llm_answer="Il CONTEXT fornito contiene una risposta sul regime ordinario.",
            corpora=[
                types.SimpleNamespace(
                    regime_id="forfettario",
                    label="Regime Forfettario",
                    path="Normativo_Forfettari_Agg_2026",
                ),
                types.SimpleNamespace(
                    regime_id="ordinario",
                    label="Regime Ordinario",
                    path="Normativo_Ordinario_Agg_2026",
                ),
            ],
        )
        response = self.ask(module, "Nel regime ordinario come funziona l'IVA?")
        self.assertNotIn("regime forfettario", response.message.lower())
        self.assertNotIn("riformula la domanda", response.message.lower())

    def test_forfettario_typos_resolve_to_default_regime(self):
        module = self.load_module()
        active_regime, regime_explicit, regime_ambiguous = module._resolve_regime(
            "Cos'è il regime forchettario?"
        )
        self.assertIsNotNone(active_regime)
        self.assertEqual(active_regime.regime_id, "forfettario")
        self.assertTrue(regime_explicit)
        self.assertFalse(regime_ambiguous)

    def test_forfettario_common_typos_are_detected(self):
        module = self.load_module()
        self.assertTrue(module._is_forfettario_query("Aliquota del regime forchettario"))
        self.assertTrue(module._is_forfettario_query("Posso accedere al regime forfetario?"))
        self.assertTrue(module._is_forfettario_query("Come funziona il regime forfetarrio"))

    def test_common_tax_term_typos_are_normalized(self):
        module = self.load_module()
        normalized = module._normalize_tax_query(
            "alliquota sogllia veis bolllo atteco contibuti partitaiva extraue"
        )
        self.assertIn("aliquota", normalized)
        self.assertIn("soglia", normalized)
        self.assertIn("vies", normalized)
        self.assertIn("bollo", normalized)
        self.assertIn("ateco", normalized)
        self.assertIn("contributi", normalized)
        self.assertIn("partita iva", normalized)
        self.assertIn("extra ue", normalized)

    def test_typo_query_routes_to_vies_answer(self):
        module = self.load_module()
        response = self.ask(
            module,
            "Nel regime forfetario il veis serve per vendere servizi in ue?",
        )
        self.assertIn("vies", response.message.lower())

    def test_typo_query_routes_to_bollo_answer(self):
        module = self.load_module()
        response = self.ask(
            module,
            "Per una fattura extraue da 120 euro devo mettere il bolllo?",
        )
        self.assertIn("bollo", response.message.lower())

    def test_typo_query_routes_to_ateco_answer(self):
        module = self.load_module()
        response = self.ask(
            module,
            "Qual è il coefficiente di redditività del codice atteco 47.82?",
        )
        self.assertIn("47.82", response.message)
        self.assertIn("%", response.message)

    def test_intro_query_returns_simple_explanation(self):
        module = self.load_module()
        response = self.ask(module, "Cos'è il regime forfettario?")
        self.assertIn("regime fiscale agevolato", response.message.lower())
        self.assertIn("15%", response.message)
        self.assertIn("5%", response.message)
        self.assertIn("coefficiente di redditività", response.message.lower())

    def test_intro_query_with_typo_returns_simple_explanation(self):
        module = self.load_module()
        response = self.ask(module, "Cos'è il regime forchettario")
        self.assertIn("regime fiscale agevolato", response.message.lower())
        self.assertNotIn("regime naturale delle persone fisiche", response.message.lower())

    def test_missing_api_key_returns_configuration_error_without_import_crash(self):
        module = self.load_module(extra_env={"API_KEY_DEEPSEEK": ""})
        response = self.ask(module, "Qual e' il limite del regime forfettario?")
        self.assertIn("API_KEY_DEEPSEEK", response.message)

    def test_semantic_search_can_be_disabled_without_loading_rag(self):
        module = self.load_module(extra_env={"SEMANTIC_SEARCH_ENABLED": "0"})

        def fail_load():
            raise AssertionError("rag.load should not be called when semantic search is disabled")

        module.rag.load = fail_load
        response = self.ask(module, "Cos'è il regime forfettario?")
        self.assertIn("regime fiscale agevolato", response.message.lower())

    def test_definition_query_returns_cited_not_defined(self):
        module = self.load_module(
            rag_results=[],
            payload_chunks=[
                {
                    "regime": "forfettario",
                    "source": "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
                    "chunk_id": 18,
                    "text": "in base al codice ATECO che contraddistingue l’attività esercitata.",
                }
            ],
            extra_env={"LEXICAL_FALLBACK_ENABLED": "1"},
        )
        response = self.ask(module, "Cos'è il codice ATECO?")
        self.assertIn("citato", response.message.lower())
        self.assertIn("non viene definito", response.message.lower())
        self.assertIn(
            "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
            response.sources,
        )

    def test_critical_questions_are_answered_correctly(self):
        module = self.load_module()
        cases = [
            (
                "Ho portato un cliente a cena fuori e ho speso 150€. Posso caricare la fattura nel software per scaricare l'IVA e abbassare le tasse del mio 15%?",
                ("non detrai", "non deduci"),
            ),
            (
                "Lavoro come dipendente e prendo 32.000€ lordi l'anno. Posso aprire la partita IVA forfettaria domani per arrotondare?",
                ("non puoi", "30.000 euro"),
            ),
            (
                "Mi sono licenziato ieri. Posso aprire la partita IVA forfettaria e fatturare tutto il mio lavoro alla mia ex azienda cosi risparmio sulle tasse?",
                ("ex datore", "anno successivo"),
            ),
            (
                "Ho venduto il mio vecchio PC aziendale usato a un amico per 400€. Devo sommare questi soldi agli 85.000€ del limite annuale?",
                ("non si somma", "85.000"),
            ),
            (
                "Ho tre figli a carico e pago 2.000€ di asilo nido. Posso recuperare il 19% di queste spese dalle tasse della mia partita IVA?",
                ("no", "imposta sostitutiva"),
            ),
            (
                "A ottobre ho incassato una super fattura e sono arrivato a 105.000€ totali nell'anno. Resto forfettario fino a dicembre e poi cambio l'anno prossimo?",
                ("uscita dal regime", "iva ordinario"),
            ),
            (
                "Ho comprato un software da un sito americano e ho pagato 100€. Non c'e l'IVA in fattura, quindi sono a posto cosi, giusto?",
                ("td17", "16 del mese successivo"),
            ),
            (
                "Ho il 20% di una SRL che si occupa di pulizie, io vorrei aprire P.IVA forfettaria per fare il consulente marketing. Posso?",
                ("sì", "20%"),
            ),
            (
                "Ho fatto una fattura da 500€ e ci ho messo la marca da bollo da 2€. Il cliente mi ha rimborsato i 2€. Su quei due euro ci devo pagare le tasse?",
                ("non costituisce un ricavo", "85.000"),
            ),
        ]

        for question, expected_terms in cases:
            with self.subTest(question=question):
                response = self.ask(module, question)
                message = response.message.lower()
                for term in expected_terms:
                    self.assertIn(term.lower(), message)

    def test_may_request_for_inps_discount_is_not_valid_for_same_year(self):
        module = self.load_module()
        response = self.ask(
            module,
            "Siamo a maggio, ho aperto la partita IVA come commerciante a gennaio ma ho pagato i contributi pieni. Posso chiedere ora lo sconto del 35% per quest'anno?",
        )
        self.assertIn("28 febbraio", response.message.lower())
        self.assertIn("anno successivo", response.message.lower())

    def test_forfettario_regression_bank(self):
        module = self.load_module()
        bank_path = Path(__file__).with_name("fixtures") / "forfettario_regression_cases.json"
        cases = json.loads(bank_path.read_text(encoding="utf-8"))
        for case in cases:
            question = case["question"]
            expected_all = case["expected_all"]
            with self.subTest(question=question):
                response = self.ask(module, question)
                message = response.message.lower()
                for term in expected_all:
                    self.assertIn(term.lower(), message)


if __name__ == "__main__":
    unittest.main()

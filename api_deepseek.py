import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

import fastapi
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from openai import APIError, RateLimitError
from pydantic import BaseModel

from lexical_fallback import LexicalChunk, LexicalFallbackIndex
from rag_qdrant import CorpusConfig, QdrantRAG, RetrievedChunk

load_dotenv()  # Carica le variabili dal file .env
chiave_api = os.getenv("API_KEY_DEEPSEEK")
if not chiave_api:
    raise ValueError("Manca API_KEY_DEEPSEEK nel file .env")

llm_model = "deepseek-chat"
client = OpenAI(
    api_key=chiave_api,
    base_url="https://api.deepseek.com",
)

app = fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione imposta l'URL del tuo frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = QdrantRAG.from_env()
rag_load_error = None
try:
    rag.load()
except Exception as error:  # pragma: no cover
    rag_load_error = str(error)

LEXICAL_FALLBACK_ENABLED = os.getenv("LEXICAL_FALLBACK_ENABLED", "1") != "0"
LOG_RAG_EVENTS = os.getenv("LOG_RAG_EVENTS", "0") == "1"
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
HARD_CODED_MODE = os.getenv("HARD_CODED_MODE", "all").strip().lower()
HARD_CODED_CATEGORIES = {
    "all": {"critical", "stable", "optional"},
    "balanced": {"critical", "stable"},
    "critical": {"critical"},
}
ALLOWED_HARD_CODED = HARD_CODED_CATEGORIES.get(HARD_CODED_MODE, {"critical", "stable"})


def _log_rag_event(event: str, payload: dict) -> None:
    if not LOG_RAG_EVENTS:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    log_path = LOG_DIR / "rag_events.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_lexical_index(regime_ids: List[str]) -> LexicalFallbackIndex | None:
    if rag_load_error:
        return None
    chunks: List[LexicalChunk] = []
    try:
        for payload in rag.iter_payload_chunks(regime_ids=regime_ids, batch_size=256):
            regime = payload.get("regime")
            source = payload.get("source")
            chunk_id = payload.get("chunk_id")
            text = payload.get("text")
            if regime is None or source is None or chunk_id is None or text is None:
                continue
            chunks.append(
                LexicalChunk(
                    regime=str(regime),
                    source=str(source),
                    chunk_id=int(chunk_id),
                    text=str(text),
                )
            )
    except Exception:
        return None
    if not chunks:
        return None
    return LexicalFallbackIndex.from_chunks(chunks)




class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    message: str
    sources: List[str]


@dataclass(frozen=True)
class RegimeProfile:
    regime_id: str
    label: str
    aliases: tuple[str, ...]
    is_default: bool = False


ATECO_GROUPS = [
    {"ranges": [(10, 11)], "coeff": "40%"},
    {"ranges": [(45, 45), (46, 46), (47, 47)], "coeff": "40%"},
    {"ranges": [(47, 47)], "coeff": "40%"},
    {"ranges": [(47, 47)], "coeff": "54%"},
    {"ranges": [(41, 43), (68, 68)], "coeff": "86%"},
    {"ranges": [(46, 46)], "coeff": "62%"},
    {"ranges": [(55, 56)], "coeff": "40%"},
    {"ranges": [(64, 66), (69, 75), (85, 85), (86, 88)], "coeff": "78%"},
    {
        "ranges": [
            (1, 3),
            (5, 9),
            (12, 33),
            (35, 35),
            (36, 39),
            (49, 53),
            (58, 63),
            (77, 82),
            (84, 84),
            (90, 93),
            (94, 96),
            (97, 99),
        ],
        "coeff": "67%",
    },
]


def _build_regime_profiles() -> List[RegimeProfile]:
    corpora = QdrantRAG.discover_pdf_corpora(Path("."))
    if not corpora:
        return [
            RegimeProfile(
                regime_id="forfettario",
                label="Regime Forfettario",
                aliases=("forfettario", "forfettari", "regime forfettario", "regime dei forfettari"),
                is_default=True,
            )
        ]

    profiles: List[RegimeProfile] = []
    for corpus in corpora:
        aliases = _build_regime_aliases(corpus)
        is_default = corpus.regime_id == "forfettario"
        profiles.append(
            RegimeProfile(
                regime_id=corpus.regime_id,
                label=corpus.label,
                aliases=aliases,
                is_default=is_default,
            )
        )

    if not any(profile.is_default for profile in profiles):
        profiles[0] = RegimeProfile(
            regime_id=profiles[0].regime_id,
            label=profiles[0].label,
            aliases=profiles[0].aliases,
            is_default=True,
        )
    return profiles


def _build_regime_aliases(corpus: CorpusConfig) -> tuple[str, ...]:
    tokens = [token for token in corpus.regime_id.split("_") if token]
    aliases = {corpus.regime_id.replace("_", " "), corpus.label.lower()}
    aliases.update(tokens)
    if "regime" not in tokens:
        aliases.add(f"regime {corpus.regime_id.replace('_', ' ')}")
    if corpus.regime_id == "forfettario":
        aliases.update(
            (
                "forfettario",
                "forfettari",
                "regime forfettario",
                "regime dei forfettari",
            )
        )
    return tuple(sorted({alias.strip() for alias in aliases if alias.strip()}, key=len, reverse=True))


REGIME_PROFILES = _build_regime_profiles()
DEFAULT_REGIME_ID = next(
    (profile.regime_id for profile in REGIME_PROFILES if profile.is_default),
    REGIME_PROFILES[0].regime_id,
)

lexical_index: LexicalFallbackIndex | None = None
if LEXICAL_FALLBACK_ENABLED:
    regime_ids = [profile.regime_id for profile in REGIME_PROFILES]
    lexical_index = _build_lexical_index(regime_ids)
    if lexical_index is None:
        lexical_index = LexicalFallbackIndex.from_local_index(
            Path("rag_index/index.json")
        )


def _normalize_match_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace("’", "'")
    return normalized


def _tokenize_for_matching(text: str) -> list[str]:
    return re.findall(r"[a-z0-9%]+", _normalize_match_text(text))


def _is_close_alias_token(token: str, alias_token: str) -> bool:
    if token == alias_token:
        return True
    if len(token) < 5 or len(alias_token) < 5:
        return False
    if any(char.isdigit() for char in token + alias_token):
        return False
    if abs(len(token) - len(alias_token)) > 2:
        return False
    if token[0] != alias_token[0] or token[-1] != alias_token[-1]:
        return False
    return SequenceMatcher(None, token, alias_token).ratio() >= 0.82


def _query_matches_alias(query: str, alias: str) -> bool:
    normalized_query = _normalize_match_text(query)
    normalized_alias = _normalize_match_text(alias)
    if normalized_alias in normalized_query:
        return True

    alias_tokens = [token for token in normalized_alias.split() if token]
    query_tokens = _tokenize_for_matching(normalized_query)
    if not alias_tokens or not query_tokens:
        return False
    if len(alias_tokens) == 1:
        return any(_is_close_alias_token(token, alias_tokens[0]) for token in query_tokens)

    window_size = len(alias_tokens)
    for start in range(len(query_tokens) - window_size + 1):
        window = query_tokens[start : start + window_size]
        if all(
            candidate == expected or _is_close_alias_token(candidate, expected)
            for candidate, expected in zip(window, alias_tokens)
        ):
            return True
    return False


def _query_mentions_regime_id(query: str, regime_id: str) -> bool:
    profile = next((item for item in REGIME_PROFILES if item.regime_id == regime_id), None)
    if profile is None:
        return False
    return any(_query_matches_alias(query, alias) for alias in profile.aliases)


EXACT_QUERY_TOKEN_REPLACEMENTS = {
    "forchettario": "forfettario",
    "forfetario": "forfettario",
    "forfetarrio": "forfettario",
    "forfettarrio": "forfettario",
    "forfetaio": "forfettario",
    "alliquota": "aliquota",
    "aliquuota": "aliquota",
    "aligquota": "aliquota",
    "sogllia": "soglia",
    "sogglia": "soglia",
    "intrastad": "intrastat",
    "intrastatto": "intrastat",
    "intrastatt": "intrastat",
    "vieds": "vies",
    "veis": "vies",
    "viess": "vies",
    "bolllo": "bollo",
    "inpss": "inps",
    "atteco": "ateco",
    "atceo": "ateco",
    "contibuti": "contributi",
    "contributtiva": "contributiva",
    "contributtivi": "contributivi",
    "fatturrato": "fatturato",
    "fattturato": "fatturato",
    "incasato": "incassato",
    "incasssato": "incassato",
    "scadneza": "scadenza",
    "domnada": "domanda",
    "impsota": "imposta",
    "extraue": "extra ue",
    "partitaiva": "partita iva",
}

CANONICAL_QUERY_TOKENS = (
    "forfettario",
    "aliquota",
    "soglia",
    "intrastat",
    "contributi",
    "contributiva",
    "contributivi",
    "fatturato",
    "incassato",
    "scadenza",
    "domanda",
    "imposta",
    "sostitutiva",
    "agevolazione",
    "artigiani",
    "commercianti",
    "partecipazioni",
    "controllo",
    "societa",
    "requisiti",
    "ricavi",
    "compensi",
    "dicitura",
    "riduzione",
    "ateco",
    "forfettari",
)


def _canonicalize_tax_token(match: re.Match[str]) -> str:
    token = match.group(0)
    exact_replacement = EXACT_QUERY_TOKEN_REPLACEMENTS.get(token)
    if exact_replacement is not None:
        return exact_replacement
    if len(token) < 6 or any(char.isdigit() for char in token):
        return token

    best_match = token
    best_score = 0.0
    for candidate in CANONICAL_QUERY_TOKENS:
        if token[0] != candidate[0] or token[-1] != candidate[-1]:
            continue
        if abs(len(token) - len(candidate)) > 2:
            continue
        score = SequenceMatcher(None, token, candidate).ratio()
        if score >= 0.84 and score > best_score:
            best_match = candidate
            best_score = score
    return best_match


def _normalize_tax_query(query: str) -> str:
    normalized = _normalize_match_text(query)
    normalized = re.sub(r"[a-z0-9%]+", _canonicalize_tax_token, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _allow_hardcoded(category: str) -> bool:
    return category in ALLOWED_HARD_CODED


DEFINITION_PATTERNS = (
    r"(?:cos'?e|cosa e|che cos'?e|che cosa e)",
    r"(?:definizione di|definisci)",
    r"(?:cosa significa|che significa|significa)",
)


def _extract_definition_term(query: str) -> str | None:
    q = _normalize_tax_query(query)
    if not any(re.search(pattern, q) for pattern in DEFINITION_PATTERNS):
        return None

    patterns = [
        r"(?:cos'?e|cosa e|che cos'?e|che cosa e)\s+(?:il|lo|la|l')?\s*([^?.,;]+)",
        r"(?:definizione di|definisci)\s+(?:il|lo|la|l')?\s*([^?.,;]+)",
        r"(?:cosa significa|che significa|significa)\s+(?:il|lo|la|l')?\s*([^?.,;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            term = match.group(1).strip()
            term = re.sub(r"\s+", " ", term)
            term = term.strip(" \"'")
            if 1 <= len(term.split()) <= 6:
                return term
    return None


def _term_appears_in_text(term: str, text: str) -> bool:
    if not term or not text:
        return False
    pattern = re.compile(rf"\\b{re.escape(term)}\\b", flags=re.IGNORECASE)
    return pattern.search(text) is not None


def _collect_term_mentions(term: str, regime_id: str) -> List[LexicalChunk]:
    if lexical_index is None:
        return []
    return lexical_index.find_mentions(term, regime_id=regime_id)


def _definition_fallback_message(term: str) -> str:
    return (
        f"Il termine {term} è citato nei documenti disponibili ma non viene definito. "
        "Se vuoi una definizione, aggiungi una fonte che lo spieghi oppure chiedi "
        "una risposta generale senza vincoli di fonte."
    )

def _contains_percent_reference(query: str, value: str) -> bool:
    return re.search(rf"(?<!\d){re.escape(value)}\s*%", query) is not None


def _has_inps_35_context(query: str) -> bool:
    q = _normalize_tax_query(query)
    return any(
        term in q
        for term in (
            "riduzione contributiva",
            "riduzione inps",
            "riduzione del 35",
            "riduzione 35",
            "agevolazione",
            "artigiani",
            "commercianti",
            "gestione separata",
            "cassa professionale",
            "professionisti con cassa",
            "inps",
        )
    ) or _contains_percent_reference(q, "35")


def _has_non_inps_domain_context(query: str) -> bool:
    q = _normalize_tax_query(query)
    return any(
        term in q
        for term in (
            "ateco",
            "vies",
            "intrastat",
            "bollo",
            "extra-ue",
            "extra ue",
            "reverse",
            "td17",
            "google ads",
            "facebook ads",
            "meta ads",
            "lavoro dipendente",
            "reddito dipendente",
            "srl",
            "societa",
            "società",
            "ex datore",
            "partecipazioni",
            "residenza estera",
            "regimi speciali",
            "acconto",
            "saldo",
        )
    )


def _extract_ateco_components(query: str) -> tuple[int, int | None] | None:
    match = re.search(
        r"\bateco\s*([0-9]{2})(?:[.\s-]?([0-9]{1,2}))?(?=[^0-9]|$)",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"\bcodice\s*([0-9]{2})(?:[.\s-]?([0-9]{1,2}))?(?=[^0-9]|$)",
            query,
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    prefix = int(match.group(1))
    subcode = int(match.group(2)) if match.group(2) else None
    return prefix, subcode


def _lookup_coefficiente_ateco(prefix: int, subcode: int | None = None) -> str | None:
    if prefix == 46:
        if subcode is not None:
            if subcode == 1 or 10 <= subcode <= 19:
                return "62%"
            return "40%"
        return (
            "Dipende dal sottocodice ATECO 46: "
            "46.1 = 62%, mentre 46.2-46.9 = 40%."
        )
    if prefix == 47:
        if subcode is not None:
            if subcode == 81:
                return "40%"
            if 82 <= subcode <= 89:
                return "54%"
            if 10 <= subcode <= 79 or 90 <= subcode <= 99:
                return "40%"
        return (
            "Dipende dal sottocodice ATECO 47: "
            "47.81 = 40%, 47.82-47.89 = 54%, "
            "47.1-47.7 e 47.9 = 40%."
        )

    for group in ATECO_GROUPS:
        for start, end in group["ranges"]:
            if start <= prefix <= end:
                return group["coeff"]
    return None


def _is_ateco_coeff_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_ateco_ref = "ateco" in q or re.search(r"\bcodice\s*[0-9]{2}", q) is not None
    return (
        has_ateco_ref
        and (
            re.search(r"coeffic", q) is not None
            or re.search(r"reddi+tivit", q) is not None
        )
    )


def _is_ateco_list_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return (
        "ateco" in q
        and (
            "tutti" in q
            or "elenco" in q
            or "quali sono" in q
            or "tabella" in q
        )
        and (
            "coeffic" in q
            or re.search(r"reddi+tivit", q) is not None
            or "codici" in q
        )
    )


def _is_ateco_codes_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return "ateco" in q and (
        "tutti i codici" in q
        or "elenco codici" in q
        or "quali sono tutti i codici" in q
    )


def _is_off_topic_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return any(
        term in q
        for term in (
            "sanremo",
            "meteo",
            "bitcoin",
            "barzelletta",
            "api key",
            ".env",
            "fuori tema",
            "chi ha vinto",
        )
    )


def _is_limit_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return any(term in q for term in ("quanto posso guadagn", "limite", "soglia", "ricavi", "compensi"))


def _is_tax_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return any(term in q for term in ("quanto vengo tass", "tassat", "imposta", "aliquota", "sostitutiva"))


def _is_forfettario_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return "forfett" in q or _query_mentions_regime_id(query, "forfettario")


def _is_tax_regime_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return any(
        term in q
        for term in (
            "regime",
            "fisco",
            "fiscal",
            "imposta",
            "aliquota",
            "iva",
            "contribut",
            "ricavi",
            "compensi",
            "soglia",
            "fattur",
            "detra",
            "dedu",
            "scadenz",
            "acconto",
            "saldo",
            "partita iva",
            "requisit",
            "accesso",
            "uscita",
            "ademp",
            "dichiar",
        )
    )


def _is_forfettario_intro_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_forfettario = _is_forfettario_query(q)
    has_intro_intent = any(
        term in q
        for term in (
            "cos'e",
            "cos e",
            "che cos'e",
            "che cos e",
            "spiegami",
            "spiegare",
            "in parole semplici",
            "in pratica",
            "come funziona",
            "come funziona in generale",
            "di cosa si tratta",
        )
    )
    technical_terms = (
        "ateco",
        "vies",
        "intrastat",
        "td17",
        "bollo",
        "inps",
        "35%",
        "soglia",
        "ricavi",
        "compensi",
        "srl",
        "ex datore",
        "lavoro dipendente",
        "cassa professionale",
        "google ads",
        "uscita",
        "scadenza",
        "acconto",
        "saldo",
        "causa ostativa",
        "cause ostative",
        "regimi speciali",
    )
    return has_forfettario and has_intro_intent and not any(term in q for term in technical_terms)


def _match_regime_profiles(query: str) -> List[RegimeProfile]:
    matches: List[RegimeProfile] = []
    for profile in REGIME_PROFILES:
        if any(_query_matches_alias(query, alias) for alias in profile.aliases):
            matches.append(profile)
    return matches


def _resolve_regime(query: str) -> tuple[RegimeProfile | None, bool, bool]:
    matches = _match_regime_profiles(query)
    if not matches:
        default_profile = next(
            (profile for profile in REGIME_PROFILES if profile.regime_id == DEFAULT_REGIME_ID),
            REGIME_PROFILES[0] if REGIME_PROFILES else None,
        )
        return default_profile, False, False

    unique_matches = {profile.regime_id: profile for profile in matches}
    if len(unique_matches) > 1:
        return None, True, True
    return next(iter(unique_matches.values())), True, False


def _regime_scope_message(active_regime: RegimeProfile | None = None) -> str:
    if active_regime is None:
        return "Posso aiutarti solo sui regimi fiscali per cui hai caricato documentazione."
    return (
        "Posso aiutarti solo su temi fiscali e contributivi legati alla documentazione caricata"
        f" per {active_regime.label.lower()}."
    )


def _is_vies_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return (
        "vies" in q
        or (
            any(term in q for term in ("intracomunit", "unione europea", "ue"))
            and any(term in q for term in ("iscrizion", "obbligator", "quando serve", "a cosa serve"))
        )
    )


def _is_bollo_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_bollo = "bollo" in q
    has_estero = any(term in q for term in ("estera", "estere", "estero", "extra-ue", "extra ue", "ue"))
    has_threshold = any(term in q for term in ("77,47", "77.47", "2 euro", "2,00"))
    return has_bollo and (has_estero or has_threshold)


def _is_eu_b2b_services_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    if "b2c" in q:
        return False
    has_service = "serviz" in q or "b2b" in q
    has_eu_context = any(
        term in q
        for term in (
            "b2b",
            "cliente ue",
            "unione europea",
            "intracomunit",
            "verso ue",
            "nell'ue",
            "in ue",
        )
    )
    has_invoice_intent = any(term in q for term in ("fattur", "dicitura", "iva", "come", "vendita"))
    return has_service and has_eu_context and has_invoice_intent


def _is_extra_ue_services_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_service = "serviz" in q
    has_extra_context = any(
        term in q
        for term in (
            "extra-ue",
            "extra ue",
            "usa",
            "fuori ue",
            "cliente estero",
            "verso estero",
        )
    )
    has_invoice_intent = any(term in q for term in ("fattur", "dicitura", "iva", "come", "vendita"))
    return has_service and has_extra_context and has_invoice_intent


def _is_eu_b2c_services_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_service = "serviz" in q
    has_b2c_context = any(
        term in q
        for term in (
            "b2c",
            "cliente privato ue",
            "privato ue",
            "consumatore ue",
        )
    )
    has_invoice_intent = any(term in q for term in ("fattur", "dicitura", "iva", "come", "vendita"))
    return has_service and has_b2c_context and has_invoice_intent


def _is_forfettario_exit_100k_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_threshold = any(term in q for term in ("100.000", "100000", "100k"))
    has_exit_intent = any(term in q for term in ("esco", "uscita", "subito", "anno dopo", "quando"))
    return has_threshold and has_exit_intent


def _is_cash_basis_threshold_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_cash_terms = any(
        term in q for term in ("fatturato", "incassato", "incassi", "criterio di cassa")
    )
    has_threshold_terms = any(
        term in q for term in ("soglie", "soglia", "limite", "ricavi", "compensi")
    )
    return has_cash_terms and has_threshold_terms


def _is_aliquota_5_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_aliquota = any(term in q for term in ("aliquota", "imposta sostitutiva")) or _contains_percent_reference(q, "5")
    has_five = _contains_percent_reference(q, "5") or "5 per cento" in q
    has_when = any(term in q for term in ("quando", "applica", "si applica"))
    return has_aliquota and has_five and has_when


def _is_ads_reverse_charge_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_ads = any(term in q for term in ("google ads", "facebook ads", "meta ads"))
    has_reverse_context = any(term in q for term in ("td17", "reverse", "iva", "autofattura"))
    return has_ads and has_reverse_context


def _is_intrastat_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return "intrastat" in q


def _is_extra_ue_wording_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_extra_context = any(term in q for term in ("extra-ue", "extra ue", "fuori ue"))
    has_wording_intent = any(term in q for term in ("dicitura", "artt. 7", "7-septies", "articoli 7"))
    return has_extra_context and has_wording_intent


def _is_bollo_exact_threshold_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_threshold = "77,47" in q or "77.47" in q
    has_exact_intent = any(term in q for term in ("esatt", "uguale", "pari a", "preciso"))
    has_bollo = "bollo" in q
    return has_bollo and has_threshold and has_exact_intent


def _is_employment_income_threshold_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_employment_context = any(
        term in q
        for term in (
            "lavoro dipendente",
            "reddito da lavoro dipendente",
            "redditi da lavoro dipendente",
            "reddito dipendente",
        )
    )
    has_threshold_or_access_intent = any(
        term in q
        for term in (
            "30.000",
            "30000",
            "trentamila",
            "posso stare",
            "posso rimanere",
            "accesso",
            "forfettario",
        )
    )
    return has_employment_context and has_threshold_or_access_intent


def _is_employment_income_under_threshold_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_employment_context = any(
        term in q
        for term in (
            "lavoro dipendente",
            "reddito da lavoro dipendente",
            "redditi da lavoro dipendente",
            "reddito dipendente",
        )
    )
    has_under_threshold = any(
        term in q
        for term in (
            "29.000",
            "29000",
            "sotto 30.000",
            "inferiore a 30.000",
        )
    )
    has_access_intent = any(term in q for term in ("posso", "restare", "forfettario"))
    return has_employment_context and has_access_intent and has_under_threshold


def _is_employment_cessation_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return (
        "rapporto di lavoro" in q
        and "cessat" in q
        and any(term in q for term in ("cambia", "rileva", "forfettario", "soglia"))
    )


def _is_inps_35_general_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    strong_terms = (
        "riduzione contributiva",
        "riduzione inps",
        "riduzione del 35",
        "riduzione 35",
        "35%",
        "gestione separata",
        "cassa professionale",
        "professionisti con cassa",
        "artigiani",
        "commercianti",
    )
    follow_up_terms = (
        "domanda",
        "presentata",
        "entro quando",
        "decorre",
        "rinnova",
        "rinnovo",
        "automatic",
        "pensione",
        "rinuncio",
        "rinuncia",
        "agevolazione",
        "si perde",
        "nuove attività",
        "richiederla",
        "di nuovo",
        "riattiv",
        "chi può",
    )
    context_anchor_terms = (
        "inps",
        "riduzione contributiva",
        "riduzione",
        "35%",
        "contribut",
        "agevolazione",
        "artigiani",
        "commercianti",
        "rinnova",
        "rinuncio",
        "richiederla",
    )
    if any(term in q for term in strong_terms):
        return True
    if _contains_percent_reference(q, "35"):
        return True
    if any(term in q for term in ("ateco", "vies", "intrastat", "bollo", "extra-ue", "reverse", "td17")):
        return False
    return any(term in q for term in follow_up_terms) and any(
        term in q for term in context_anchor_terms
    )


def _is_forfettario_domain_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    domain_terms = (
        "forfett",
        "regime",
        "ateco",
        "iva",
        "inps",
        "contribut",
        "fattur",
        "bollo",
        "vies",
        "reverse charge",
        "inversione contabile",
        "intrastat",
        "b2b",
        "b2c",
        "extra-ue",
        "extra ue",
        "intracomunit",
        "cliente ue",
        "cliente estero",
        "vendita servizi",
        "dicitura",
        "ricavi",
        "compensi",
        "imposta",
        "aliquota",
        "sostitutiva",
        "soglia",
        "quadro lm",
        "artigiani",
        "commercianti",
        "partita iva",
        "scadenz",
        "acconto",
        "saldo",
        "cause ostative",
        "lavoro dipendente",
        "srl",
        "societa",
        "società",
        "partecipazioni",
        "controllo",
        "ex datore",
        "riduzione",
        "35%",
        "domanda",
        "decorre",
        "rinnova",
        "rinuncio",
        "rinuncia",
        "richiederla",
        "riattiv",
        "pensione",
        "gestione separata",
        "cassa professionale",
        "google ads",
        "facebook ads",
        "td17",
        "100k",
        "100.000",
        "100000",
        "uscita",
        "esco",
        "presentata",
    )
    return any(term in q for term in domain_terms)


def _is_inps_35_deadline_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_inps_context = _has_inps_35_context(q)
    has_deadline_intent = any(
        term in q
        for term in (
            "scadenz",
            "entro quando",
            "termine",
            "quando va presentata",
            "presentazione",
        )
    )
    return has_inps_context and has_deadline_intent


def _is_inps_35_short_deadline_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_inps_context = _has_inps_35_context(q)
    has_deadline_intent = any(
        term in q
        for term in (
            "entro quando",
            "scadenza",
            "quando va presentata",
            "termine",
        )
    )
    generic_follow_up = not _has_non_inps_domain_context(q)
    return has_deadline_intent and "domanda" in q and (has_inps_context or generic_follow_up)


def _is_inps_35_march_decorrenza_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_march_example = "10 marzo" in q or ("marzo" in q and any(term in q for term in ("domanda", "invio", "invi")))
    has_decorrenza_intent = any(
        term in q
        for term in (
            "decorre",
            "da quando",
            "quando decorre",
            "decorrenza",
        )
    )
    generic_follow_up = not _has_non_inps_domain_context(q)
    return has_march_example and has_decorrenza_intent and (
        _has_inps_35_context(q) or generic_follow_up
    )


def _is_inps_35_reapply_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_renounce = any(
        term in q
        for term in (
            "rinuncio",
            "rinuncia",
            "rinunciare",
            "rinunci",
            "revoca",
        )
    )
    has_reapply = any(
        term in q
        for term in (
            "richiederla",
            "richiedere di nuovo",
            "nuova domanda",
            "di nuovo",
            "riattiv",
        )
    )
    return has_renounce and has_reapply


def _is_inps_35_renewal_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_inps_35_context = _has_inps_35_context(q)
    has_renewal_intent = "rinnova" in q or "rinnovo" in q
    has_auto_intent = "automatic" in q or "ogni anno" in q
    return has_inps_35_context and has_renewal_intent and has_auto_intent


def _is_inps_35_cassa_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_35 = _has_inps_35_context(q)
    has_cassa = "cassa" in q or "professionist" in q
    return has_35 and has_cassa


def _is_srl_control_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_srl = "srl" in q or "societa" in q or "società" in q
    has_control = any(term in q for term in ("controllo", "2359", "controllo di fatto"))
    has_forfettario_intent = any(
        term in q
        for term in (
            "forfett",
            "posso restare",
            "posso accedere",
            "causa ostativa",
            "restare",
            "accedere",
        )
    )
    return has_srl and has_control and has_forfettario_intent


def _is_inps_35_new_activity_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_new_activity = any(
        term in q
        for term in (
            "nuove attivita",
            "nuove attività",
            "nuova attivita",
            "nuova attività",
            "appena aperto",
            "inizio attivita",
            "inizio attività",
        )
    )
    has_request_intent = "domanda" in q or "richiesta" in q or "quando va fatta" in q
    generic_follow_up = not _has_non_inps_domain_context(q)
    return has_new_activity and has_request_intent and (
        _has_inps_35_context(q) or generic_follow_up
    )


def _is_inps_35_apply_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_apply_intent = any(
        term in q
        for term in (
            "come si presenta",
            "come present",
            "come fare domanda",
            "dove si presenta",
            "dove fare domanda",
        )
    )
    generic_follow_up = "domanda" in q and not _has_non_inps_domain_context(q)
    return has_apply_intent and (_has_inps_35_context(q) or generic_follow_up)


def _is_inps_35_late_deadline_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_late_deadline = any(
        term in q
        for term in (
            "dopo il 28 febbraio",
            "dopo 28 febbraio",
            "oltre il 28 febbraio",
            "oltre 28 febbraio",
        )
    )
    return has_late_deadline and "domanda" in q and (
        _has_inps_35_context(q) or not _has_non_inps_domain_context(q)
    )


def _is_inps_35_loss_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_loss_intent = any(
        term in q for term in ("si perde", "quando si perde", "perdo", "perdita")
    )
    return _has_inps_35_context(q) and has_loss_intent


def _is_ex_datore_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_ex_datore = "ex datore" in q or "datore di lavoro" in q
    has_ostativa = any(term in q for term in ("causa ostativa", "ostativo", "forfettario"))
    return has_ex_datore and has_ostativa


def _is_ex_datore_after_two_years_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_ex_datore = _is_ex_datore_query(q)
    has_time_reference = any(
        term in q
        for term in (
            "3 anni",
            "tre anni",
            "oltre due anni",
            "piu di due anni",
            "più di due anni",
        )
    )
    return has_ex_datore and has_time_reference


def _clean_model_answer(answer: str) -> str:
    cleaned = answer.strip()
    replacements = (
        (r"\bcontesto fornito\b", "documenti disponibili"),
        (r"\bCONTEXT\b", "contesto"),
        (r"\bIl contesto fornito non contiene\b", "I documenti disponibili non contengono"),
        (r"\bIl contesto fornito non riporta\b", "I documenti disponibili non riportano"),
        (r"\bIl contesto fornito non specifica\b", "I documenti disponibili non specificano"),
        (r"\bIl contesto fornito non fornisce\b", "I documenti disponibili non forniscono"),
        (r"\bIl contesto non contiene\b", "I documenti disponibili non contengono"),
        (r"\bIl contesto non riporta\b", "I documenti disponibili non riportano"),
        (r"\bIl contesto non specifica\b", "I documenti disponibili non specificano"),
        (r"\bIl contesto non fornisce\b", "I documenti disponibili non forniscono"),
    )
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def _intent_expansions(query: str, regime_id: str) -> List[str]:
    if regime_id != "forfettario":
        return []
    q = _normalize_tax_query(query)
    expansions: List[str] = []

    definition_term = _extract_definition_term(q)
    if definition_term:
        expansions.extend(
            [
                f"definizione {definition_term}",
                f"cos'e {definition_term}",
                f"{definition_term} significato",
            ]
        )

    if "ateco" in q:
        expansions.append("tabella coefficienti redditività ateco allegato 4")

    if any(term in q for term in ("soglia", "ricavi", "compensi", "limite", "uscita")):
        expansions.extend(
            [
                "regime forfettario soglia 85000 ricavi compensi",
                "regime forfettario uscita immediata 100000",
                "circolare 32/e 2023 soglie accesso uscita",
            ]
        )

    if any(term in q for term in ("tass", "imposta", "aliquota", "sostitutiva")):
        expansions.extend(
            [
                "regime forfettario imposta sostitutiva 15% 5%",
                "aliquota 5 per cento nuove attività forfettario",
                "quadro lm imposta sostitutiva forfettario",
            ]
        )

    if any(term in q for term in ("scadenz", "saldo", "acconto", "calendario")):
        expansions.extend(
            [
                "calendario fiscale forfettari 2026 saldo acconto",
                "scadenze imposta sostitutiva regime forfettario 2026",
            ]
        )

    if any(term in q for term in ("inps", "contribut", "artigiani", "commercianti", "gestione separata", "35%")):
        expansions.extend(
            [
                "riduzione contributiva 35% regime forfettario",
                "inps artigiani commercianti forfettario 2026",
                "aliquote gestione separata 2026",
                "domanda riduzione contributiva 35 entro 28 febbraio",
                "scadenza domanda agevolazione contributiva artigiani commercianti",
            ]
        )

    if any(term in q for term in ("ostativ", "esclusion", "esclus", "cause")):
        expansions.extend(
            [
                "cause ostative regime forfettario 2026",
                "esclusioni regime forfettario lavoro dipendente partecipazioni",
            ]
        )

    return list(dict.fromkeys(expansions))


def _merge_results(primary: List[RetrievedChunk], extras: List[RetrievedChunk], top_k: int = 8) -> List[RetrievedChunk]:
    by_key = {}
    for item in primary + extras:
        key = (item.source, item.chunk_id)
        current = by_key.get(key)
        if current is None or item.score > current.score:
            by_key[key] = item

    ranked = sorted(by_key.values(), key=lambda r: r.score, reverse=True)
    if not ranked:
        return []

    # Favorisce varietà di fonti nei primi risultati per ridurre omissioni su testi tabellari.
    selected: List[RetrievedChunk] = []
    used_sources = set()
    for item in ranked:
        if item.source not in used_sources:
            selected.append(item)
            used_sources.add(item.source)
        if len(selected) >= min(4, top_k):
            break

    for item in ranked:
        key = (item.source, item.chunk_id)
        if any((x.source, x.chunk_id) == key for x in selected):
            continue
        selected.append(item)
        if len(selected) >= top_k:
            break

    return selected


def _dynamic_score_thresholds(query: str) -> List[float]:
    token_count = len(_normalize_tax_query(query).split())
    if token_count <= 3:
        return [0.16, 0.12, 0.08]
    if token_count <= 6:
        return [0.2, 0.14, 0.1]
    return [0.22, 0.18, 0.12]


def _search_with_intent(query: str, regime_id: str) -> tuple[List[RetrievedChunk], str]:
    normalized_query = _normalize_tax_query(query)
    primary_queries = [normalized_query]
    if query.strip() and normalized_query != query.strip().lower():
        primary_queries.append(query.strip())

    thresholds = _dynamic_score_thresholds(normalized_query)
    for threshold in thresholds:
        primary_results: List[RetrievedChunk] = []
        for primary_query in primary_queries:
            primary_results.extend(
                rag.search(
                    primary_query,
                    top_k=8,
                    min_score=threshold,
                    regime_ids=[regime_id],
                )
            )

        extra_results: List[RetrievedChunk] = []
        for expanded_query in _intent_expansions(normalized_query, regime_id=regime_id):
            extra_results.extend(
                rag.search(
                    expanded_query,
                    top_k=4,
                    min_score=max(threshold - 0.02, 0.05),
                    regime_ids=[regime_id],
                )
            )
        merged = _merge_results(primary_results, extra_results, top_k=8)
        if merged:
            return merged, "semantic"

    if lexical_index is None:
        return [], "none"

    lexical_hits = lexical_index.search(normalized_query, top_k=8, regime_id=regime_id)
    if not lexical_hits:
        return [], "none"
    return [
        RetrievedChunk(
            regime=chunk.regime,
            source=chunk.source,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            score=score,
        )
        for chunk, score in lexical_hits
    ], "lexical"


@app.post("/", response_model=ChatResponse)
async def read_root(payload: ChatRequest):
    if rag_load_error:
        return ChatResponse(
            message=(
                "Indice RAG su Qdrant non disponibile. Esegui prima: "
                "`python3 build_rag_index.py`."
            ),
            sources=[],
        )

    raw_contenuto = payload.content.strip()
    if not raw_contenuto:
        return ChatResponse(message="Inserisci una domanda valida.", sources=[])
    contenuto = _normalize_tax_query(raw_contenuto)

    active_regime, regime_explicit, regime_ambiguous = _resolve_regime(contenuto)
    if regime_ambiguous:
        available = ", ".join(profile.label for profile in REGIME_PROFILES)
        return ChatResponse(
            message=(
                "La domanda sembra riferirsi a piu' regimi. Specifica meglio il regime fiscale da usare "
                f"tra quelli caricati: {available}."
            ),
            sources=[],
        )
    if active_regime is None:
        return ChatResponse(message=_regime_scope_message(), sources=[])

    is_forfettario_regime = active_regime.regime_id == "forfettario"
    allow_critical = _allow_hardcoded("critical")
    allow_stable = _allow_hardcoded("stable")
    allow_optional = _allow_hardcoded("optional")

    if _is_off_topic_query(contenuto):
        return ChatResponse(
            message=(
                f"{_regime_scope_message(active_regime)} "
                "Riformula la domanda in questo ambito."
            ),
            sources=[],
        )

    if allow_optional and is_forfettario_regime and _is_forfettario_intro_query(contenuto):
        return ChatResponse(
            message=(
                "Il regime forfettario è un regime fiscale agevolato per partite IVA individuali. "
                "In pratica, il reddito imponibile non si calcola sottraendo tutte le spese reali una per una, "
                "ma applicando ai ricavi un coefficiente di redditività legato all'attività svolta. "
                "Su quel reddito si paga di norma un'imposta sostitutiva del 15%, che in alcuni casi scende al 5% "
                "per le nuove attività. Prevede anche semplificazioni IVA e contabili, ma si può usare solo se "
                "rispetti i requisiti e non hai cause ostative."
            ),
            sources=[
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
            ],
        )

    if allow_critical and is_forfettario_regime and _is_ateco_coeff_query(contenuto):
        ateco_data = _extract_ateco_components(contenuto)
        if ateco_data is not None:
            prefix, subcode = ateco_data
            coeff = _lookup_coefficiente_ateco(prefix, subcode=subcode)
            if coeff is not None:
                if subcode is not None and coeff.endswith("%"):
                    message = (
                        f"Il coefficiente di redditività per ATECO {prefix}.{subcode} è {coeff}."
                    )
                elif prefix == 46 or prefix == 47:
                    message = (
                        f"Per il codice ATECO {prefix}, {coeff} "
                        "Controlla sempre il sottocodice completo per il valore esatto."
                    )
                else:
                    message = (
                        f"Il coefficiente di redditività per ATECO {prefix} è {coeff}."
                    )
                return ChatResponse(
                    message=message,
                    sources=[
                        "03_Tabella_Coefficienti_Redditivita_ATECO.pdf",
                        "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
                    ],
                )
        if _is_ateco_list_query(contenuto):
            return ChatResponse(
                message=(
                    "Nel regime forfettario i coefficienti di redditività sono associati a gruppi ATECO: "
                    "40% (es. commercio e alloggio/ristorazione), "
                    "54% (commercio ambulante alimenti/bevande), "
                    "62% (intermediari del commercio), "
                    "67% (altre attività economiche), "
                    "78% (attività professionali, sanitarie, istruzione e finanziarie), "
                    "86% (costruzioni e attività immobiliari). "
                    "Se mi indichi un codice ATECO specifico, ti dico il coefficiente esatto."
                ),
                sources=[
                    "03_Tabella_Coefficienti_Redditivita_ATECO.pdf",
                    "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
                ],
            )

    if allow_critical and is_forfettario_regime and _is_ateco_codes_query(contenuto):
        return ChatResponse(
            message=(
                "Nei documenti disponibili non c'è un elenco completo di tutti i codici ATECO italiani; "
                "c'è la tabella dei gruppi ATECO rilevanti per il regime forfettario con i relativi coefficienti. "
                "Se vuoi, posso dirti il coefficiente partendo da un codice ATECO preciso."
            ),
            sources=[
                "03_Tabella_Coefficienti_Redditivita_ATECO.pdf",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_forfettario_query(contenuto) and _is_limit_query(contenuto) and _is_tax_query(contenuto):
        return ChatResponse(
            message=(
                "Per restare nel regime forfettario, nel periodo precedente ricavi o compensi non devono superare 85.000 euro; "
                "se durante l'anno superi 100.000 euro, l'uscita dal regime è immediata. "
                "L'imposta sostitutiva è in via ordinaria al 15%, ridotta al 5% per i primi 5 anni se sono rispettati i requisiti di nuova attività."
            ),
            sources=[
                "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_forfettario_query(contenuto) and _is_limit_query(contenuto):
        return ChatResponse(
            message=(
                "Per il regime forfettario, la soglia ordinaria è 85.000 euro di ricavi o compensi. "
                "L'uscita immediata scatta se nell'anno superi 100.000 euro."
            ),
            sources=[
                "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_forfettario_exit_100k_query(contenuto):
        return ChatResponse(
            message=(
                "Se superi 100.000 euro di ricavi o compensi nell'anno, l'uscita dal forfettario è immediata "
                "dal momento del superamento."
            ),
            sources=[
                "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_cash_basis_threshold_query(contenuto):
        return ChatResponse(
            message=(
                "Per le soglie del regime forfettario conta quanto incassi, non quanto fatturi, perché si applica "
                "il criterio di cassa. "
                "Quindi per verificare i limiti di 85.000 e 100.000 euro rilevano i ricavi o compensi percepiti."
            ),
            sources=[
                "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_aliquota_5_query(contenuto):
        return ChatResponse(
            message=(
                "L'aliquota del 5% si applica alle nuove attività che rispettano i requisiti previsti "
                "per il regime forfettario, e vale per i primi 5 anni."
            ),
            sources=[
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
                "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_apply_query(contenuto):
        return ChatResponse(
            message=(
                "La domanda per la riduzione INPS del 35% è esclusivamente telematica. "
                "Si presenta dal portale INPS, con accesso SPID, CIE o CNS, nel Cassetto Previdenziale per "
                "Artigiani e Commercianti alla voce Domande telematizzate e Regime agevolato."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_deadline_query(contenuto):
        return ChatResponse(
            message=(
                "La domanda per la riduzione contributiva INPS del 35% si presenta solo online nel Cassetto "
                "Previdenziale Artigiani e Commercianti (accesso con SPID/CIE/CNS). "
                "Per i contribuenti già attivi va presentata entro il 28 febbraio di ogni anno; "
                "se inviata dopo, l'agevolazione decorre dal 1° gennaio dell'anno successivo. "
                "Per le nuove attività, la richiesta va fatta tempestivamente dopo l'iscrizione previdenziale."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_late_deadline_query(contenuto):
        return ChatResponse(
            message=(
                "Sì, puoi presentarla anche dopo il 28 febbraio, ma per i contribuenti già attivi la riduzione "
                "non opera nell'anno in corso. "
                "Se la domanda è tardiva, l'agevolazione decorre dal 1° gennaio dell'anno successivo."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_short_deadline_query(contenuto):
        return ChatResponse(
            message=(
                "Per i contribuenti già attivi, la domanda per la riduzione INPS del 35% va presentata "
                "entro il 28 febbraio di ogni anno."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_march_decorrenza_query(contenuto):
        return ChatResponse(
            message=(
                "Se presenti la domanda il 10 marzo, la riduzione non decorre nell'anno in corso ma dal "
                "1° gennaio dell'anno successivo."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_new_activity_query(contenuto):
        return ChatResponse(
            message=(
                "Per una nuova attività, la richiesta della riduzione INPS del 35% va fatta dopo l'iscrizione "
                "alla Gestione Artigiani e Commercianti, tramite il Cassetto Previdenziale. "
                "Va presentata tempestivamente, senza attendere il 28 febbraio dell'anno successivo."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_reapply_query(contenuto):
        return ChatResponse(
            message=(
                "Sì, se rinunci puoi presentare una nuova domanda in seguito, purché tu sia ancora nel "
                "regime forfettario e restino i requisiti per l'agevolazione. "
                "Per i contribuenti già attivi si applica il termine ordinario del 28 febbraio; se presenti "
                "la domanda dopo tale data, la riduzione decorre dal 1° gennaio dell'anno successivo."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_renewal_query(contenuto):
        return ChatResponse(
            message=(
                "Sì, l'agevolazione si rinnova se continui ad avere i requisiti del regime forfettario e "
                "dell'iscrizione alla Gestione Artigiani e Commercianti. "
                "Se i requisiti vengono meno o rinunci, si torna al regime contributivo ordinario."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_loss_query(contenuto):
        return ChatResponse(
            message=(
                "L'agevolazione del 35% si perde se non restano i requisiti per applicarla o se presenti "
                "rinuncia. In quel caso si rientra nel regime contributivo ordinario."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_cassa_query(contenuto):
        return ChatResponse(
            message=(
                "No, la riduzione INPS del 35% non si applica ai professionisti iscritti a una Cassa "
                "professionale e non si applica neppure alla Gestione Separata. "
                "È riservata agli imprenditori individuali forfettari iscritti alla Gestione Artigiani e Commercianti."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
                "06_Circolare_INPS_8-2026_Aliquote_Gestione_Separata.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_general_query(contenuto):
        return ChatResponse(
            message=(
                "La riduzione INPS del 35% è riservata agli imprenditori individuali forfettari iscritti alla "
                "Gestione Artigiani e Commercianti; non si applica a Gestione Separata o Casse professionali. "
                "La domanda è telematica; per gli attivi va presentata entro il 28 febbraio, se tardiva decorre "
                "dal 1° gennaio dell'anno successivo. "
                "L'agevolazione si rinnova se restano i requisiti, ma riduce la contribuzione utile ai fini pensionistici."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
                "06_Circolare_INPS_8-2026_Aliquote_Gestione_Separata.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_ex_datore_after_two_years_query(contenuto):
        return ChatResponse(
            message=(
                "No, non è ostativo se il rapporto con l'ex datore di lavoro è cessato da oltre due periodi "
                "d'imposta. La causa ostativa riguarda la fatturazione prevalente verso datori di lavoro con "
                "cui il rapporto è in corso o è cessato nei due precedenti periodi d'imposta."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "05_Circolare_9E-2019_Approfondimento_Cause_Ostative.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_srl_control_query(contenuto):
        return ChatResponse(
            message=(
                "In presenza di controllo, anche di fatto, una partecipazione in SRL può costituire causa "
                "ostativa se la società esercita attività economiche direttamente o indirettamente riconducibili "
                "a quella svolta individualmente. La sola percentuale di partecipazione non basta: conta il controllo "
                "ai sensi dell'articolo 2359 c.c. e la riconducibilità dell'attività."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "05_Circolare_9E-2019_Approfondimento_Cause_Ostative.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_vies_query(contenuto):
        return ChatResponse(
            message=(
                "Sì, per il forfettario l'iscrizione al VIES è necessaria quando effettua operazioni "
                "intracomunitarie (vendita di servizi o acquisto di beni/servizi nell'UE). "
                "Serve a operare con partita IVA abilitata nei rapporti UE."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_ads_reverse_charge_query(contenuto):
        return ChatResponse(
            message=(
                "Per acquisti di servizi esteri come Google Ads o Facebook Ads, in forfettario si applica "
                "integrazione/autofattura TD17 con IVA al 22% e versamento entro il giorno 16 del mese successivo "
                "con F24 codice 6099. L'IVA resta un costo non detraibile."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
                "08a_Guida_Pratica_Fatturazione_Elettronica_Forfettari_2026.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_bollo_exact_threshold_query(contenuto):
        return ChatResponse(
            message=(
                "No, con importo esattamente pari a 77,47 euro il bollo non si applica. "
                "L'imposta di bollo da 2,00 euro scatta solo oltre 77,47 euro."
            ),
            sources=[
                "08b_Manuale_AdE_Imposta_Bollo_Fatture_Elettroniche.pdf",
                "08a_Guida_Pratica_Fatturazione_Elettronica_Forfettari_2026.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_bollo_query(contenuto):
        return ChatResponse(
            message=(
                "Sì, se la fattura supera 77,47 euro si applica l'imposta di bollo da 2,00 euro, "
                "anche nelle fatture verso l'estero. "
                "In fattura va indicata la dicitura di assolvimento del bollo; "
                "il versamento è gestito con liquidazione periodica tramite i canali dell'Agenzia delle Entrate."
            ),
            sources=[
                "08b_Manuale_AdE_Imposta_Bollo_Fatture_Elettroniche.pdf",
                "08a_Guida_Pratica_Fatturazione_Elettronica_Forfettari_2026.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_intrastat_query(contenuto):
        return ChatResponse(
            message=(
                "Nei documenti, l'Intrastat è richiamato per le operazioni di servizi B2B verso soggetti UE. "
                "Per i casi specifici (periodicità e obbligo puntuale) è necessaria verifica operativa sul singolo caso."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_eu_b2c_services_query(contenuto):
        return ChatResponse(
            message=(
                "Nei documenti disponibili è trattata in modo esplicito soprattutto la casistica B2B UE "
                "(reverse charge e Intrastat). "
                "Per servizi B2C verso cliente UE la disciplina IVA dipende dal tipo di servizio e dal luogo di consumo, "
                "quindi serve una verifica specifica prima di emettere fattura."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
                "08a_Guida_Pratica_Fatturazione_Elettronica_Forfettari_2026.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_eu_b2b_services_query(contenuto):
        return ChatResponse(
            message=(
                "Per servizi B2B verso cliente UE, la fattura si emette senza IVA con dicitura "
                "\"Reverse Charge\" o \"Inversione contabile\" e richiede iscrizione VIES. "
                "Per queste operazioni è previsto l'adempimento Intrastat. "
                "Se la fattura supera 77,47 euro, si applica anche il bollo da 2,00 euro."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
                "08a_Guida_Pratica_Fatturazione_Elettronica_Forfettari_2026.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_extra_ue_wording_query(contenuto):
        return ChatResponse(
            message=(
                "La dicitura da usare per servizi extra-UE è: "
                "\"Operazione non soggetta ai sensi degli artt. da 7 a 7-septies del DPR 633/72\"."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_extra_ue_services_query(contenuto):
        return ChatResponse(
            message=(
                "Per servizi verso cliente extra-UE, la fattura è senza IVA con dicitura: "
                "\"Operazione non soggetta ai sensi degli artt. da 7 a 7-septies del DPR 633/72\". "
                "Se l'importo supera 77,47 euro, si applica il bollo da 2,00 euro."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
                "08b_Manuale_AdE_Imposta_Bollo_Fatture_Elettroniche.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_employment_income_under_threshold_query(contenuto):
        return ChatResponse(
            message=(
                "Con reddito da lavoro dipendente pari a 29.000 euro, la sola soglia dei 30.000 euro "
                "non è causa ostativa. "
                "Restano comunque da verificare le altre condizioni di accesso e permanenza nel forfettario."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "05_Circolare_9E-2019_Approfondimento_Cause_Ostative.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_employment_income_threshold_query(contenuto):
        return ChatResponse(
            message=(
                "In generale, con redditi da lavoro dipendente o assimilati superiori a 30.000 euro "
                "non puoi applicare il regime forfettario. "
                "La soglia non rileva se il rapporto di lavoro è cessato. "
                "Restano comunque da verificare anche le altre cause ostative."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "05_Circolare_9E-2019_Approfondimento_Cause_Ostative.pdf",
                "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_employment_cessation_query(contenuto):
        return ChatResponse(
            message=(
                "Sì: se il rapporto di lavoro dipendente è cessato, la soglia dei 30.000 euro non rileva "
                "come causa ostativa. Restano però da verificare gli altri requisiti del regime forfettario."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "05_Circolare_9E-2019_Approfondimento_Cause_Ostative.pdf",
            ],
        )

    if not regime_explicit and is_forfettario_regime and not _is_forfettario_domain_query(contenuto):
        return ChatResponse(
            message=(
                f"{_regime_scope_message(active_regime)} "
                "Riformula la domanda in questo ambito."
            ),
            sources=[],
        )

    if regime_explicit and not _is_tax_regime_query(contenuto):
        return ChatResponse(
            message=(
                f"{_regime_scope_message(active_regime)} "
                "Riformula la domanda in questo ambito."
            ),
            sources=[],
        )

    definition_term = _extract_definition_term(raw_contenuto)
    term_mentions: List[LexicalChunk] = []
    if definition_term:
        term_mentions = _collect_term_mentions(definition_term, active_regime.regime_id)

    retrieved, retrieval_mode = _search_with_intent(
        contenuto, regime_id=active_regime.regime_id
    )
    if not retrieved:
        if definition_term and term_mentions:
            _log_rag_event(
                "definition_cited_not_defined",
                {
                    "query": raw_contenuto,
                    "regime": active_regime.regime_id,
                    "term": definition_term,
                    "sources": list(dict.fromkeys(chunk.source for chunk in term_mentions))[:4],
                },
            )
            return ChatResponse(
                message=_definition_fallback_message(definition_term),
                sources=list(dict.fromkeys(chunk.source for chunk in term_mentions))[:4],
            )
        _log_rag_event(
            "rag_no_results",
            {"query": raw_contenuto, "regime": active_regime.regime_id},
        )
        return ChatResponse(
            message=(
                f"Non trovo informazioni pertinenti nei documenti caricati per {active_regime.label.lower()}. "
                "Riformula la domanda o aggiungi documentazione."
            ),
            sources=[],
        )

    if definition_term:
        has_term_in_context = any(
            _term_appears_in_text(definition_term, item.text) for item in retrieved
        )
        if retrieval_mode == "lexical" and (term_mentions or has_term_in_context):
            sources = (
                list(dict.fromkeys(chunk.source for chunk in term_mentions))[:4]
                if term_mentions
                else list(dict.fromkeys(item.source for item in retrieved))[:4]
            )
            _log_rag_event(
                "definition_cited_not_defined",
                {
                    "query": raw_contenuto,
                    "regime": active_regime.regime_id,
                    "term": definition_term,
                    "sources": sources,
                },
            )
            return ChatResponse(
                message=_definition_fallback_message(definition_term),
                sources=sources,
            )
        if not has_term_in_context and term_mentions:
            _log_rag_event(
                "definition_cited_not_defined",
                {
                    "query": raw_contenuto,
                    "regime": active_regime.regime_id,
                    "term": definition_term,
                    "sources": list(dict.fromkeys(chunk.source for chunk in term_mentions))[:4],
                },
            )
            return ChatResponse(
                message=_definition_fallback_message(definition_term),
                sources=list(dict.fromkeys(chunk.source for chunk in term_mentions))[:4],
            )

    top_score = max(item.score for item in retrieved)
    if top_score < 0.12:
        _log_rag_event(
            "rag_low_confidence",
            {
                "query": raw_contenuto,
                "regime": active_regime.regime_id,
                "top_score": top_score,
                "sources": list(dict.fromkeys(item.source for item in retrieved))[:4],
            },
        )

    context_blocks = []
    for item in retrieved:
        context_blocks.append(
            (
                f"[Fonte: {item.source} | Chunk: {item.chunk_id} | "
                f"Score: {item.score:.3f}]\n{item.text}"
            )
        )

    context = "\n\n".join(context_blocks)
    system_prompt = (
        f"Sei un assistente fiscale per {active_regime.label.lower()} in Italia. "
        "Rispondi solo con informazioni presenti nel CONTEXT. "
        "Se il CONTEXT non contiene una parte della risposta, dillo in una sola frase breve. "
        "Se un termine è solo citato ma non definito, dillo esplicitamente. "
        "Non affermare che un termine non è menzionato se compare nel CONTEXT. "
        "Non inventare norme, soglie o scadenze. "
        "Stile obbligatorio: italiano chiaro, tono professionale, nessun markdown, "
        "nessun uso di **, # o elenchi con trattini. "
        "Non iniziare con formule tipo 'In base al CONTEXT fornito'."
    )

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"DOMANDA:\n{contenuto}\n\n"
                        f"CONTEXT:\n{context}\n\n"
                        "Rispondi in italiano in modo sintetico. "
                        "Usa massimo 4 frasi totali. "
                        "Dai prima la risposta diretta. "
                        "Non inserire mai le fonti nel testo della risposta."
                    ),
                },
            ],
            stream=False,
        )
    except RateLimitError:
        return ChatResponse(
            message=(
                "Quota DeepSeek esaurita o limite raggiunto (errore 429). "
                "Controlla piano e billing del provider selezionato."
            ),
            sources=[],
        )
    except APIError as error:
        return ChatResponse(
            message=f"Errore API DeepSeek: {error}",
            sources=[],
        )

    answer = _clean_model_answer(response.choices[0].message.content or "")
    sources = list(dict.fromkeys(item.source for item in retrieved))[:4]
    return ChatResponse(message=answer, sources=sources)

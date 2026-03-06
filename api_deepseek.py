import os
import re
from typing import List

import fastapi
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from openai import APIError, RateLimitError
from pydantic import BaseModel

from rag_qdrant import QdrantRAG, RetrievedChunk

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


class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    message: str
    sources: List[str]


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


def _extract_ateco_components(query: str) -> tuple[int, int | None] | None:
    match = re.search(
        r"\bateco\s*([0-9]{2})(?:[.\s-]?([0-9]{1,2}))?\b",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"\bcodice\s*([0-9]{2})(?:[.\s-]?([0-9]{1,2}))?\b",
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
    q = query.lower()
    return (
        "ateco" in q
        and (
            re.search(r"coeffic", q) is not None
            or re.search(r"reddi+tivit", q) is not None
        )
    )


def _is_limit_query(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in ("quanto posso guadagn", "limite", "soglia", "ricavi", "compensi"))


def _is_tax_query(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in ("quanto vengo tass", "tassat", "imposta", "aliquota", "sostitutiva"))


def _is_forfettario_query(query: str) -> bool:
    return "forfett" in query.lower()


def _is_vies_query(query: str) -> bool:
    q = query.lower()
    return (
        "vies" in q
        or (
            any(term in q for term in ("intracomunit", "unione europea", "ue"))
            and any(term in q for term in ("iscrizion", "obbligator", "quando serve", "a cosa serve"))
        )
    )


def _is_bollo_query(query: str) -> bool:
    q = query.lower()
    has_bollo = "bollo" in q
    has_estero = any(term in q for term in ("estera", "estere", "estero", "extra-ue", "extra ue", "ue"))
    has_threshold = any(term in q for term in ("77,47", "77.47", "2 euro", "2,00"))
    return has_bollo and (has_estero or has_threshold)


def _is_eu_b2b_services_query(query: str) -> bool:
    q = query.lower()
    if "b2c" in q:
        return False
    has_service = "serviz" in q
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
    q = query.lower()
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
    q = query.lower()
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


def _is_bollo_exact_threshold_query(query: str) -> bool:
    q = query.lower()
    has_threshold = "77,47" in q or "77.47" in q
    has_exact_intent = any(term in q for term in ("esatt", "uguale", "pari a", "preciso"))
    has_bollo = "bollo" in q
    return has_bollo and has_threshold and has_exact_intent


def _is_employment_income_threshold_query(query: str) -> bool:
    q = query.lower()
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


def _is_forfettario_domain_query(query: str) -> bool:
    q = query.lower()
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
    )
    return any(term in q for term in domain_terms)


def _is_inps_35_deadline_query(query: str) -> bool:
    q = query.lower()
    has_inps_context = any(
        term in q
        for term in (
            "inps",
            "riduzione contributiva",
            "35%",
            "artigiani",
            "commercianti",
            "agevolazione",
        )
    )
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


def _intent_expansions(query: str) -> List[str]:
    q = query.lower()
    expansions: List[str] = []

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


def _search_with_intent(query: str) -> List[RetrievedChunk]:
    primary = rag.search(query, top_k=8, min_score=0.2)
    extra_results: List[RetrievedChunk] = []
    for expanded_query in _intent_expansions(query):
        extra_results.extend(rag.search(expanded_query, top_k=4, min_score=0.18))
    return _merge_results(primary, extra_results, top_k=8)


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

    contenuto = payload.content.strip()
    if not contenuto:
        return ChatResponse(message="Inserisci una domanda valida.", sources=[])

    if _is_ateco_coeff_query(contenuto):
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

    if _is_forfettario_query(contenuto) and _is_limit_query(contenuto) and _is_tax_query(contenuto):
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

    if _is_inps_35_deadline_query(contenuto):
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

    if _is_vies_query(contenuto):
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

    if _is_bollo_exact_threshold_query(contenuto):
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

    if _is_bollo_query(contenuto):
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

    if _is_eu_b2c_services_query(contenuto):
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

    if _is_eu_b2b_services_query(contenuto):
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

    if _is_extra_ue_services_query(contenuto):
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

    if _is_employment_income_threshold_query(contenuto):
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

    if not _is_forfettario_domain_query(contenuto):
        return ChatResponse(
            message=(
                "Posso aiutarti solo su temi fiscali e contributivi del regime forfettario. "
                "Riformula la domanda in questo ambito."
            ),
            sources=[],
        )

    retrieved = _search_with_intent(contenuto)
    if not retrieved:
        return ChatResponse(
            message=(
                "Non trovo informazioni pertinenti nei documenti forniti. "
                "Riformula la domanda o aggiungi documentazione."
            ),
            sources=[],
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
        "Sei un assistente fiscale per il regime forfettario italiano. "
        "Rispondi solo con informazioni presenti nel CONTEXT. "
        "Se il CONTEXT non contiene una parte della risposta, dillo in una sola frase breve. "
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

    answer = response.choices[0].message.content or ""
    sources = list(dict.fromkeys(item.source for item in retrieved))
    return ChatResponse(message=answer, sources=sources)

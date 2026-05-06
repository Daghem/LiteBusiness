---
title: FlyTax
emoji: "🧾"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
fullWidth: true
header: default
short_description: Assistente AI e simulatore per il regime forfettario.
---

# Spiegazione progetto

---

## 1. Cos'è FlyTax

**FlyTax** è un assistente AI specializzato sul **regime forfettario italiano**. Non è un chatbot generico: risponde solo a domande fiscali e contributive legate a questo regime, usando come fonte una biblioteca di documenti ufficiali (leggi, circolari, guide tecniche, prassi dell'Agenzia delle Entrate e INPS).

Il cuore è un sistema **RAG** (Retrieval Augmented Generation): l'AI non risponde "a memoria", ma legge i documenti normativi caricati, trova i passaggi pertinenti, e poi compone la risposta.

---

## 2. Architettura e tecnologie

Il progetto è un'applicazione web full-stack con questi strati:

| Strato | Tecnologia | File principali |
|---|---|---|
| **Frontend** | HTML, CSS, Bootstrap 5, JavaScript vanilla | [index.html](cci:7://file:///home/ytaki/its/project_work/FlyTax/index.html:0:0-0:0), [chat.html](cci:7://file:///home/ytaki/its/project_work/FlyTax/chat.html:0:0-0:0), [dashboard.html](cci:7://file:///home/ytaki/its/project_work/FlyTax/dashboard.html:0:0-0:0), [admin_tools.html](cci:7://file:///home/ytaki/its/project_work/FlyTax/admin_tools.html:0:0-0:0) |
| **Backend API** | FastAPI (Python) | [api_deepseek.py](cci:7://file:///home/ytaki/its/project_work/FlyTax/api_deepseek.py:0:0-0:0) |
| **AI / LLM** | DeepSeek (via API compatibile OpenAI) | [api_deepseek.py](cci:7://file:///home/ytaki/its/project_work/FlyTax/api_deepseek.py:0:0-0:0) |
| **Database vettoriale** | Qdrant | [rag_qdrant.py](cci:7://file:///home/ytaki/its/project_work/FlyTax/rag_qdrant.py:0:0-0:0) |
| **Embedding** | SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`) | [rag_qdrant.py](cci:7://file:///home/ytaki/its/project_work/FlyTax/rag_qdrant.py:0:0-0:0) |
| **Simulatore fiscale** | Python puro | [tax_simulator.py](cci:7://file:///home/ytaki/its/project_work/FlyTax/tax_simulator.py:0:0-0:0) |
| **Storage dati** | File JSON/JSONL su disco | [storage_services.py](cci:7://file:///home/ytaki/its/project_work/FlyTax/storage_services.py:0:0-0:0) |
| **Indicizzazione** | Script CLI | [build_rag_index.py](cci:7://file:///home/ytaki/its/project_work/FlyTax/build_rag_index.py:0:0-0:0) |

---

## 3. Come funziona la chat (flusso RAG)

Quando scrivi una domanda in chat, il backend esegue questi passaggi:

### A. Normalizzazione e controllo ambito
- La domanda viene "ripulita" da errori di battitura comuni (`forchettario` → `forfettario`, `alliquota` → `aliquota`, ecc.).
- Il sistema controlla se la domanda è **off-topic** (es. "che tempo fa?", "bitcoin") e la rifiuta.
- Verifica se la domanda rientra nel dominio del forfettario.

### B. Risposte hardcoded (guscio di precisione)
Prima di chiamare l'AI, il backend ha una **lunga serie di regole codificate** che riconoscono domande specifiche e rispondono immediatamente con testo controllato. Esempi:
- Coefficienti ATECO
- Soglie 85.000€ / 100.000€
- Aliquota 5% e 15%
- Riduzione INPS 35%
- Fatturazione verso UE / extra-UE
- Cause ostative (lavoro dipendente, ex datore, SRL)
- NASPI e compatibilità
- Bollo, VIES, Intrastat

Questo serve a **non inventare numeri o norme**: se la domanda è una di queste, la risposta è precisa al 100% senza passare dall'AI.

### C. Ricerca nei documenti (RAG)
Se non c'è una regola hardcoded, il sistema:
1. **Embedda la domanda** (la trasforma in un vettore numerico con SentenceTransformers).
2. **Cerca su Qdrant** i chunk di testo più simili tra i PDF/XML indicizzati.
3. Usa anche un **fallback lessicale** (ricerca per parole chiave) se la ricerca semantica trova poco.
4. **Espande la query**: aggiunge termini correlati per trovare più fonti (es. se chiedi di "tasse", cerca anche "imposta sostitutiva", "aliquota", "quadro LM").

### D. Generazione della risposta con DeepSeek
- I chunk trovati vengono incollati in un **prompt strutturato** con istruzioni rigide: *"Rispondi solo con informazioni presenti nel CONTEXT. Non inventare norme. Stile: italiano chiaro, tono professionale, nessun markdown, massimo 4 frasi."*
- DeepSeek legge i documenti e compone la risposta.
- La risposta viene pulita (es. toglie frasi tipo "In base al contesto fornito").

### E. Fonti e confidenza
La risposta arriva all'utente con:
- **Fonti**: nomi dei PDF usati, con estratto del testo e numero di pagina (quando disponibile).
- **Confidenza**: `alta`, `media` o `bassa`, in base allo score dei risultati trovati.

---

## 4. Il simulatore fiscale

In [chat.html](cci:7://file:///home/ytaki/its/project_work/FlyTax/chat.html:0:0-0:0) e [dashboard.html](cci:7://file:///home/ytaki/its/project_work/FlyTax/dashboard.html:0:0-0:0) c'è un modulo per calcolare in modo **orientativo**:
- **Imponibile**: ricavi × coefficiente di redditività (lookup da codice ATECO).
- **Contributi**: imponibile × aliquota previdenziale (Gestione Separata, Artigiani/Commercianti, o nessuna).
- **Imposta sostitutiva**: 15% o 5% sull'imponibile meno i contributi.
- **Netto stimato**.

I coefficienti ATECO sono codificati in [tax_simulator.py](cci:7://file:///home/ytaki/its/project_work/FlyTax/tax_simulator.py:0:0-0:0) e in [api_deepseek.py](cci:7://file:///home/ytaki/its/project_work/FlyTax/api_deepseek.py:0:0-0:0) (duplicati per uso in chat e in simulatore).

---

## 5. Gestione documentale e indicizzazione

I documenti normativi stanno in [Normativo_Forfettari_Agg_2026/](cci:9://file:///home/ytaki/its/project_work/FlyTax/Normativo_Forfettari_Agg_2026:0:0-0:0) (PDF e XML).

### Indicizzazione ([build_rag_index.py](cci:7://file:///home/ytaki/its/project_work/FlyTax/build_rag_index.py:0:0-0:0) + [rag_qdrant.py](cci:7://file:///home/ytaki/its/project_work/FlyTax/rag_qdrant.py:0:0-0:0))
1. Estrae il testo da ogni pagina PDF (con PyMuPDF / `fitz`).
2. Divide il testo in **chunk** di ~1200 caratteri con sovrapposizione di 200.
3. Genera gli **embedding** (vettori) per ogni chunk.
4. Carica tutto su **Qdrant** con metadata: regime, nome file, chunk_id, testo, pagina iniziale/finale.

In runtime, la ricerca è istantanea: Qdrant confronta il vettore della domanda con i vettori dei chunk.

---

## 6. Frontend: le 3 pagine

- **[index.html](cci:7://file:///home/ytaki/its/project_work/FlyTax/index.html:0:0-0:0)**: landing page di presentazione con call-to-action.
- **[chat.html](cci:7://file:///home/ytaki/its/project_work/FlyTax/chat.html:0:0-0:0)**: workspace vero e proprio. Ha:
  - Pannello laterale con cronologia chat salvate.
  - Area messaggi con possibilità di dare feedback (👍/👎).
  - Simulatore forfettario collassabile.
  - Toggle tema chiaro/scuro.
- **[dashboard.html](cci:7://file:///home/ytaki/its/project_work/FlyTax/dashboard.html:0:0-0:0)**: pagina di benvenuto con accesso rapido a chat, simulatore e admin.
- **[admin_tools.html](cci:7://file:///home/ytaki/its/project_work/FlyTax/admin_tools.html:0:0-0:0)**: area tecnica per upload PDF/XML, re-indicizzazione e statistiche.

Tutte le pagine usano Bootstrap 5, font moderni (Manrope, Space Grotesk) e un tema scuro/claro persistente in `localStorage`.

---

## 7. Backend API: endpoint principali ([api_deepseek.py](cci:7://file:///home/ytaki/its/project_work/FlyTax/api_deepseek.py:0:0-0:0))

| Endpoint | Metodo | Cosa fa |
|---|---|---|
| `POST /` | Chat sincrona | Domanda → risposta JSON con testo, fonti, confidenza. |
| `POST /chat-stream` | Chat streaming | Stessa logica ma risponde in SSE (Server-Sent Events) per effetto "macchina da scrivere". |
| `POST /simulate` | Simulatore | Calcola tasse e contributi. |
| `GET /regimes` | Regimi supportati | Restituisce solo `forfettario`. |
| `POST /chat-history` | Salva turno | Memorizza la conversazione su disco. |
| `GET /chat-history` | Lista chat | Elenco cronologia. |
| `GET /chat-history/{id}` | Recupera chat | Riapre una chat salvata. |
| `DELETE /chat-history/{id}` | Elimina chat | |
| `POST /feedback` | Salva feedback | Up/down sulla risposta. |
| `GET /admin/overview` | Statistiche | Numero chat, messaggi, feedback, domande top. |
| `POST /admin/upload` | Carica PDF/XML | Richiede chiave admin. |
| `POST /admin/reindex` | Ricostruisce indice | Rilegge tutti i PDF e ricarica Qdrant. |
| `GET /healthz` | Health check | Stato del servizio e del RAG. |

---

## 8. Storage e persistenza

Tutto è basato su **file JSON/JSONL** (nessun database relazionale):
- `data/chat_history/*.json` — ogni chat è un file JSON con i messaggi.
- `data/feedback/feedback.jsonl` — una riga per ogni voto.
- `data/events/app_events.jsonl` — log di eventi (RAG senza risultati, confidenza bassa, reindex).

Le statistiche admin vengono calcolate al volo leggendo questi file.
---

## 10. Deploy

Il progetto è pronto per essere deployato su:
- **Hugging Face Spaces** (Docker Space, porta 7860).
- **Render** (con [render.yaml](cci:7://file:///home/ytaki/its/project_work/FlyTax/render.yaml:0:0-0:0) come Blueprint).
- **Locale** con `uvicorn api_deepseek:app --reload`.

Il [Dockerfile](cci:7://file:///home/ytaki/its/project_work/FlyTax/Dockerfile:0:0-0:0) è incluso, così come [space_server.py](cci:7://file:///home/ytaki/its/project_work/FlyTax/space_server.py:0:0-0:0) per l'avvio su HF Spaces.

---

## In sintesi

FlyTax è un **sistema RAG specializzato e blindato** per il regime forfettario. La sua forza è la combinazione di:
1. **Risposte hardcoded** per le domande frequenti (massima precisione).
2. **Ricerca semantica + lessicale** su documenti ufficiali (Qdrant).
3. **LLM controllato** (DeepSeek) che risponde solo leggendo le fonti trovate, con prompt che vietano di inventare.
4. **Simulatore fiscale** integrato per i calcoli orientativi.

È pensato per studi professionali, consulenti o privati che vogliono risposte affidabili sul forfettario senza rischio di "allucinazioni" dell'AI.
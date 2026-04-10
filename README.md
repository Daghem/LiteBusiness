# FlyTax

## Progetto ITS

Applicazione web con assistente AI per regimi fiscali italiani.
Il retrieval documentale usa Qdrant (database vettoriale) con indicizzazione diretta dai PDF normativi.

## Stack

- Frontend statico: `index.html`, `chat.html`
- Dashboard operativa: `admin.html`
- Backend: FastAPI (`api_deepseek.py`)
- LLM: DeepSeek via API compatibile OpenAI
- Vector DB: Qdrant
- Embedding: SentenceTransformers

## Variabili ambiente

Nel file `.env`:

```env
API_KEY_DEEPSEEK=...
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=flytax_normativa_2026
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LEXICAL_FALLBACK_ENABLED=1
HARD_CODED_MODE=all
LOG_RAG_EVENTS=0
LOG_DIR=logs
DATA_ROOT=data
DOCUMENT_ROOTS=.
UPLOADS_ROOT=.
RAG_INDEX_PATH=rag_index/index.json
ADMIN_ACCESS_KEY=...
```

## Avvio rapido

1. Avvia Qdrant:
   `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`
2. Installa dipendenze:
   `pip install -r requirements.txt`
3. Indicizza i PDF (e XML):
   `python3 build_rag_index.py`
4. Avvia API:
   `uvicorn api_deepseek:app --reload`
5. Apri `http://127.0.0.1:8000/` dal browser.
6. Dashboard utente: `http://127.0.0.1:8000/admin.html`
7. Area admin tecnica: `http://127.0.0.1:8000/admin_tools.html`

## Deploy su Render

Il repository include `render.yaml`, quindi puo' essere importato come Blueprint.

Configurazione minima:

1. Crea un servizio Qdrant raggiungibile da Render.
2. Su Render importa il repo con `render.yaml`.
3. Imposta almeno queste variabili:
   `API_KEY_DEEPSEEK`, `QDRANT_URL`, `QDRANT_API_KEY` (se serve), `ADMIN_ACCESS_KEY`.
4. Avvia il servizio web.

Note pratiche per Render:

- L'app espone sia API sia frontend dallo stesso processo FastAPI.
- La root `GET /` serve `index.html`, mentre `POST /` resta l'endpoint chat.
- Se vuoi usare la stessa origin per il frontend deployato, non serve modificare i file HTML: usano automaticamente `window.location.origin`.
- Per avere persistenza reale su chat, feedback, log e documenti caricati, conviene montare un disco Render e puntare gli env `DATA_ROOT`, `UPLOADS_ROOT`, `LOG_DIR` e, se necessario, `DOCUMENT_ROOTS` verso quel mount path.

## Funzionalita' aggiunte

- Chat con cronologia persistita lato server e lista chat recenti
- Selettore esplicito del regime da interrogare
- Fonti arricchite con estratti, score e pagina quando disponibile
- Indicatore di confidenza della risposta (`alta`, `media`, `bassa`)
- Export delle risposte e feedback `utile / non utile`
- Simulatore fiscale orientativo per il regime forfettario
- Upload PDF/XML da dashboard e reindicizzazione da interfaccia
- Metriche base su chat, feedback e query senza risultato
- Test automatici su simulatore e servizi di storage

## Endpoint principali

- `POST /` chat AI
- `GET /regimes` lista corpora/regimi disponibili
- `POST /simulate` simulatore forfettario
- `GET /chat-history` lista chat salvate
- `POST /chat-history` salva un turno chat
- `GET /chat-history/{chat_id}` recupera una chat
- `DELETE /chat-history/{chat_id}` elimina una chat
- `POST /feedback` salva feedback utente
- `GET /admin/overview` statistiche dashboard
- `POST /admin/upload` carica un PDF/XML
- `POST /admin/reindex` ricostruisce l'indice Qdrant

## Note

- L'indice non usa piu' `testi_estratti_2026` per la ricerca runtime.
- I chunk testuali vengono salvati come payload su Qdrant e recuperati on-demand.
- Le cartelle `Normativo_*` vengono indicizzate automaticamente come corpora distinti (supporta `.pdf` e `.xml`, anche in sottocartelle).
- `DOCUMENT_ROOTS` accetta piu' percorsi separati da virgola, utile per combinare documenti inclusi nel repo e documenti caricati su un disco persistente.
- Ogni corpus viene associato a un `regime` nel payload Qdrant, cosi' il chatbot puo' filtrare i risultati per regime.
- Le regole hardcoded attuali restano specializzate sul regime forfettario; per altri regimi il chatbot usa il flusso RAG/LLM sui documenti caricati.
- E' attivo un fallback lessicale opzionale per evitare falsi "non menzionato" in caso di retrieval debole.
- `HARD_CODED_MODE`: `all`, `balanced`, `critical` per limitare le risposte hardcoded.
- Per domande definitorie (es. "cos'è il codice ATECO") e' consigliato aggiungere una fonte ufficiale (ISTAT/AdE) che includa la definizione.
- Per vedere le pagine nelle fonti, e' necessario reindicizzare i documenti con la versione aggiornata di `build_rag_index.py`.

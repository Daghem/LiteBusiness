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
5. Apri `chat.html` dal browser.
6. Apri `admin.html` per dashboard, upload documenti e reindex.

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
- Ogni corpus viene associato a un `regime` nel payload Qdrant, cosi' il chatbot puo' filtrare i risultati per regime.
- Le regole hardcoded attuali restano specializzate sul regime forfettario; per altri regimi il chatbot usa il flusso RAG/LLM sui documenti caricati.
- E' attivo un fallback lessicale opzionale per evitare falsi "non menzionato" in caso di retrieval debole.
- `HARD_CODED_MODE`: `all`, `balanced`, `critical` per limitare le risposte hardcoded.
- Per domande definitorie (es. "cos'è il codice ATECO") e' consigliato aggiungere una fonte ufficiale (ISTAT/AdE) che includa la definizione.
- Per vedere le pagine nelle fonti, e' necessario reindicizzare i documenti con la versione aggiornata di `build_rag_index.py`.

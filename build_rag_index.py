from pathlib import Path

from app_paths import DOCUMENT_ROOTS
from dotenv import load_dotenv

from rag_qdrant import QdrantRAG

FORFETTARIO_CORPUS_DIRNAME = "Normativo_Forfettari_Agg_2026"


def main() -> None:
    load_dotenv()
    rag = QdrantRAG.from_env()
    for root in DOCUMENT_ROOTS:
        candidate = Path(root) / FORFETTARIO_CORPUS_DIRNAME
        if not candidate.is_dir():
            continue
        if not any(candidate.rglob("*.pdf")) and not any(candidate.rglob("*.xml")):
            continue
        corpus = QdrantRAG.derive_corpus_config(candidate)
        total_chunks = rag.build_from_pdf_directories(
            corpora=[corpus],
            chunk_size=1200,
            overlap=200,
            embed_batch_size=32,
            recreate_collection=True,
        )
        print(
            "Indicizzazione completata su Qdrant: "
            f"collection={rag.collection_name}, chunk={total_chunks}, regime={corpus.regime_id}"
        )
        return

    raise FileNotFoundError(
        f"Nessuna cartella {FORFETTARIO_CORPUS_DIRNAME} con PDF/XML trovata."
    )


if __name__ == "__main__":
    main()

from pathlib import Path

from app_paths import DOCUMENT_ROOTS
from dotenv import load_dotenv

from rag_qdrant import QdrantRAG


def main() -> None:
    load_dotenv()
    rag = QdrantRAG.from_env()
    corpora = []
    for root in DOCUMENT_ROOTS:
        corpora.extend(QdrantRAG.discover_pdf_corpora(Path(root)))
    if not corpora:
        raise FileNotFoundError("Nessuna cartella Normativo_* con PDF trovata.")

    total_chunks = rag.build_from_pdf_directories(
        corpora=corpora,
        chunk_size=1200,
        overlap=200,
        embed_batch_size=32,
        recreate_collection=True,
    )
    indexed_regimes = ", ".join(corpus.regime_id for corpus in corpora)
    print(
        "Indicizzazione completata su Qdrant: "
        f"collection={rag.collection_name}, chunk={total_chunks}, regimi={indexed_regimes}"
    )


if __name__ == "__main__":
    main()

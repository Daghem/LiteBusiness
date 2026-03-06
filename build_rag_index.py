from pathlib import Path

from dotenv import load_dotenv

from rag_qdrant import QdrantRAG


def main() -> None:
    load_dotenv()
    rag = QdrantRAG.from_env()
    total_chunks = rag.build_from_pdf_directory(
        pdf_dir=Path("Normativo_Forfettari_Agg_2026"),
        chunk_size=1200,
        overlap=200,
        embed_batch_size=32,
        recreate_collection=True,
    )
    print(
        "Indicizzazione completata su Qdrant: "
        f"collection={rag.collection_name}, chunk={total_chunks}"
    )


if __name__ == "__main__":
    main()

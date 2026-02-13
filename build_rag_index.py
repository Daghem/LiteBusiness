from pathlib import Path

from rag import LocalRAG


def main() -> None:
    rag = LocalRAG(index_file=Path("rag_index/index.json"))
    rag.build_from_directory(
        text_dir=Path("testi_estratti_2026"),
        chunk_size=1200,
        overlap=200,
    )
    rag.save()
    print(f"Indice creato: {rag.index_file} (chunk: {len(rag.chunks)})")


if __name__ == "__main__":
    main()

from pathlib import Path

import fitz  # PyMuPDF


def estrai_testo_pdf(percorso_pdf: Path) -> str:
    testo_totale = []
    with fitz.open(percorso_pdf) as documento:
        for pagina in documento:
            testo_totale.append(pagina.get_text())
    return "".join(testo_totale)


def estrai_documenti_cartella(
    cartella_input: Path = Path("Normativo_Forfettari_Agg_2026"),
    cartella_output: Path = Path("testi_estratti_2026"),
) -> None:
    cartella_output.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(cartella_input.glob("*.pdf"))
    if not pdf_files:
        print(f"Nessun PDF trovato in: {cartella_input}")
        return

    file_aggregato = cartella_output / "tutti_i_documenti.txt"
    with file_aggregato.open("w", encoding="utf-8") as aggregato:
        for pdf_path in pdf_files:
            testo = estrai_testo_pdf(pdf_path)

            output_txt = cartella_output / f"{pdf_path.stem}.txt"
            output_txt.write_text(testo, encoding="utf-8")

            aggregato.write(f"\n\n===== {pdf_path.name} =====\n\n")
            aggregato.write(testo)

            print(f"Estratto: {pdf_path.name} -> {output_txt}")

    print(f"\nCompletato. File aggregato: {file_aggregato}")


if __name__ == "__main__":
    estrai_documenti_cartella()

# Eden PDF vo Qdrant — za brzo osvezuvanje bez cel run_ingestion.
# Primer: python -m ingestion.ingest_one "data/pdfs/.../fajl.pdf"
from __future__ import annotations
import sys
from pathlib import Path
from ingestion.run_ingestion import ( PDF_DIR, _load_pdf_manifest, _push, _parchinja_od_tabeli,)
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_document
from ingestion.ocr_ops import treba_pregled, zapisi_za_pregled  # PII karantin (isto kako run_ingestion)


def ingest_one(pdf_path: Path, dry_run: bool) -> None:
    pdf = pdf_path.resolve()                     # apsolutna pateka do fajlot
    if not pdf.exists():                         # dokolku fajlot ne postoi
        print(f"НЕ постои: {pdf}")
        return
    manifest = _load_pdf_manifest()              # naslov/link manifest (ako ima)
    try:
        izvor = str(pdf.relative_to(PDF_DIR.resolve()))   # izvor = relativna pateka vo data/pdfs
    except ValueError:
        izvor = pdf.name                         # ako e nadvor od data/pdfs, koristi go imeto
    meta = manifest.get(izvor, {}) or manifest.get(pdf.name, {})   # metapodatoci od manifest

    tekst = load_pdf(pdf)                         # izvlekuvanje na tekstot od pdf
    # 1A: odluki / spisoci so iminja → za_pregled, ne vo Qdrant
    if treba_pregled(izvor, tekst):
        if not dry_run:
            zapisi_za_pregled(izvor, tekst, {"title": meta.get("title") or pdf.stem})  # zapisi tekst + meta vo data/ocr/za_pregled za racen pregled
            print(f"Karantin (licni podatoci): {izvor} → data/ocr/za_pregled")
        else: # dry-run: samo najava, bez zapis na disk
            print(f"[dry-run] karantin (ne bi se zapishal): {izvor}")
        return  # stop — ne seche parchinja i ne odi vo Qdrant
    parchinja = chunk_document(                   # sechenje na parchinja
        tekst, source=izvor, doc_type="pdf",
        title=meta.get("title") or pdf.stem, url=meta.get("url"),
    )
    for parche in parchinja:
        parche.metadata.setdefault("content_type", "prose")
    if meta.get("tables", True):                 # i tabelite (ako gi ima)
        parchinja += _parchinja_od_tabeli(pdf, izvor, meta)

    try:                                         # struktura na papkite -> metadata (ciklus/fakultet/nasoka)
        delovi = pdf.relative_to(PDF_DIR.resolve()).parts[:-1]  # papkite bez imeto na fajlot
        for parche in parchinja:
            if len(delovi) >= 1:
                parche.metadata["ciklus"] = delovi[0]       # prva papka = ciklus
            if len(delovi) >= 3:
                parche.metadata["fakultet"] = delovi[2]     # treta papka = fakultet
            if len(delovi) >= 4:
                parche.metadata["nasoka"] = delovi[3]       # cetvrta papka = nasoka
    except ValueError:
        pass  # fajlot e nadvor od data/pdfs — nema struktura za metadata
    broj = _push(parchinja, izvor, dry_run)      # zapis vo Qdrant (ili proba pri dry-run)
    print(f"{izvor}: {broj} парчиња {'(dry-run, не запишани)' if dry_run else 'запишани во Qdrant'}")

if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    pdfs = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not pdfs:
        print('Дај патека до PDF. Пример:\n  python -m ingestion.ingest_one "data/pdfs/vtor ciklus/upis_vtor_2026_27.pdf"')
        sys.exit(1)
    for p in pdfs:
        ingest_one(Path(p), dry)

import pdfplumber

pdf_path = "atc_alphabetical.pdf"
all_text = []
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        print(f"\n--- Página {i+1} ---\n")
        print(text)
        all_text.append(text)

# Guarda un txt para inspeccionar
with open("atc_texto_extraido.txt", "w", encoding="utf8") as f:
    for t in all_text:
        if t:
            f.write(t + "\n\n")

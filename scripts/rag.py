import fitz  # PyMuPDF
import ollama

# PDF einlesen
def read_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# RAG‑Antwort erstellen
def rag_response(pdf_text, question):
    # Hier bauen wir eine simple prompt‑Kombination
    prompt = f"Hier ist der Text aus dem PDF:\n{pdf_text}\n\nBeantworte die Frage:\n{question}"

    # Ollama‑Chat aufrufen
    response = ollama.chat(
        model="llama2",  # z. B. llama2 oder ein anderes Modell, das du lokal hast
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    # Antworttext extrahieren
    return response["message"]["content"]

if __name__ == "__main__":
    pdf_file = "/mnt/d/rag/data/beispiel.pdf"
    question = "Was ist die Kernbotschaft dieses Dokuments?"

    text = read_pdf(pdf_file)
    answer = rag_response(text, question)
    print("\nAntwort vom Modell:\n")
    print(answer)

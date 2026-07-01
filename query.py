from rag_core import ensure_models, answer_question


def main():
    ensure_models()
    question = input("Frage: ").strip()

    if not question:
        print("Keine Frage eingegeben.")
        return

    answer, sources = answer_question(question)

    print("\nAntwort:\n")
    print(answer)

    print("\nQuellen:\n")
    for i, source in enumerate(sources, start=1):
        print(f"{i}. {source}")


if __name__ == "__main__":
    main()

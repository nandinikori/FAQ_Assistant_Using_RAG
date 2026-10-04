"""Application entrypoint for the AI FAQ Assistant.

This module is the command center for the project. It initializes the runtime
configuration, checks the database health, and then lets the user decide which
workflow to run:

1. Data ingestion: load the FAQ dataset and store it in PostgreSQL with vector
   embeddings for semantic search.
2. RAG retrieval: find and display the closest FAQ matches.
3. AI answer: ask Gemini to compose an answer using the retrieved FAQ context.
4. Exit: stop the program cleanly.

The goal of this file is to keep the operational flow simple for both local
usage and portfolio demonstration while delegating the real work to the
service layer.
"""

from pathlib import Path

from app.config.settings import get_settings
from app.infrastructure.database import DatabaseClient
from app.infrastructure.embedding_client import GeminiEmbeddingClient
from app.services.AI_assistant_service import AIAssistantService
from app.services.faq_ingestion_service import FAQIngestionService
from app.services.rag_service import RAGService

PROJECT_ROOT = Path(__file__).resolve().parent


def show_menu() -> None:
    """Display the available user actions in a simple command-line menu."""
    print("\n=== AI FAQ Assistant Menu ===")
    print("1. Ingest FAQ data into PostgreSQL")
    print("2. RAG retrieval: show matching FAQs")
    print("3. Ask AI: generate an answer from retrieved FAQs")
    print("4. Exit")
    print("=============================")


def handle_ingestion(settings: object) -> None:
    """Load the FAQ dataset from disk and insert it into the vector database."""
    # Create the database and embedding clients needed by the ingestion layer.
    database_client = DatabaseClient(settings)
    embedding_client = GeminiEmbeddingClient(settings)
    ingestion_service = FAQIngestionService(database_client, embedding_client)

    print("\nPreparing FAQ ingestion...")

    # Check that PostgreSQL is actually available before attempting inserts.
    if not database_client.ping():
        print("Database is not reachable. Start PostgreSQL and ensure pgvector is enabled before ingestion.")
        return

    try:
        faq_file = PROJECT_ROOT / "data" / "faqs.json"
        if not faq_file.exists():
            raise FileNotFoundError(f"The FAQ source file was not found at {faq_file}")

        faqs = ingestion_service.load_json(faq_file)
        print(f"Loaded {len(faqs)} FAQ records from the source file.")

        inserted_count = ingestion_service.ingest(faqs)
        print(f"Successfully ingested {inserted_count} FAQ records into the knowledge base.")
    except FileNotFoundError as exc:
        print(f"The FAQ source file was not found: {exc}")
    except Exception as exc:
        print(f"Ingestion failed: {exc}")


def handle_retrieval(settings: object) -> None:
    """Retrieve and display FAQ matches without generating an AI answer."""
    rag_service = RAGService(settings)

    question = input("\nEnter your question: ").strip()
    if not question:
        question = "How can computers understand the meaning of text?"

    try:
        matches = rag_service.retrieve(question)
        if not matches:
            print("No matching FAQs were found.")
            return

        print("\nRetrieved FAQ matches:")
        for rank, match in enumerate(matches, start=1):
            print(f"\n{rank}. {match['question']}")
            print(f"   Answer: {match['answer']}")
            if match.get("category"):
                print(f"   Category: {match['category']}")
            if match.get("similarity") is not None:
                print(f"   Similarity: {float(match['similarity']):.3f}")
    except Exception as exc:
        print(f"Retrieval failed: {exc}")


def handle_ai_answer(settings: object) -> None:
    """Generate a formatted AI answer grounded in retrieved FAQ matches."""
    assistant = AIAssistantService(settings)
    question = input("\nEnter your question: ").strip()
    if not question:
        question = "How can computers understand the meaning of text?"

    try:
        answer = assistant.ask_ai(question)
        print("\nAI:")
        print(answer)
    except Exception as exc:
        print(f"AI answer failed: {exc}")


def main() -> None:
    """Run the interactive assistant until the user chooses to exit."""
    # Load environment-based configuration and validate required secrets.
    settings = get_settings()

    # Initialize the database client once so the menu can report connectivity.
    database_client = DatabaseClient(settings)

    print("AI FAQ Assistant starting...")
    print(f"Database target: {settings.db_host}/{settings.db_name}")

    if database_client.ping():
        print("Database connection successful.")
    else:
        print("Database connection failed. PostgreSQL may not be running or pgvector may not be configured.")

    # The application keeps running in a loop so the user can decide what action
    # to perform next without restarting the script each time.
    while True:
        show_menu()
        choice = input("Select an option [1-4]: ").strip().lower()

        if choice == "1":
            handle_ingestion(settings)
        elif choice == "2":
            handle_retrieval(settings)
        elif choice == "3":
            handle_ai_answer(settings)
        elif choice in {"4", "exit", "quit"}:
            print("Goodbye! Thanks for using the AI FAQ Assistant.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()

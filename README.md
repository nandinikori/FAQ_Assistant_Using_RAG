# AI FAQ Assistant

A modular PostgreSQL + pgvector + Gemini RAG project for semantic FAQ search and retrieval.

This project stores FAQ records in PostgreSQL, generates embeddings with Gemini, and performs similarity search using pgvector so users can query the knowledge base by meaning rather than exact keyword match.

<a id="contents"></a>
## Contents and day-by-day progress

| Section | Overview / progress |
| --- | --- |
| [Why this project exists](#why-this-project-exists) | Motivation for semantic FAQ search. |
| [Features](#features) | Main project capabilities. |
| [Tech stack](#tech-stack) | Languages, services, and libraries used. |
| [Project structure](#project-structure) | Repository layout. |
| [Prerequisites](#prerequisites) | Required software and credentials. |
| [Local setup](#local-setup) | Installation and configuration steps. |
| [Run the application](#run-the-application) | How to start the CLI. |
| [Data flow](#data-flow) | High-level RAG pipeline. |
| [Testing](#testing) | Test suite and how to run it. |
| [Notes for public GitHub use](#notes-for-public-github-use) | Credential and repository guidance. |
| [Development progress](#development-progress) — [Day 1](#day-1) | **Basic FAQ RAG baseline:** JSON ingestion, Gemini embeddings, PostgreSQL/pgvector semantic retrieval, and context-based answer generation through a CLI. |
| [Day 2 — Document ingestion and chunking](#day-2-progress) | **Progress:** versioned synthetic policies, seven chunking strategies, provenance, and offline structural comparison. |

## Why this project exists

Traditional keyword search fails when users ask questions with different wording than the source content. This assistant uses embeddings to convert both the FAQ and the user question into vectors and then finds the closest semantic matches in PostgreSQL.

[Back to index](#contents)

## Features

- FAQ dataset ingestion from JSON into PostgreSQL
- Gemini embedding generation for each FAQ and user question
- Gemini-generated, Markdown-formatted answers grounded in retrieved FAQs
- pgvector similarity search using cosine distance
- modular Python package structure
- CLI for ingesting data and asking questions
- document parsing and chunking strategy comparison with source provenance
- production-oriented project layout for portfolio and GitHub use

[Back to index](#contents)

## Tech stack

- Python 3.13+
- PostgreSQL
- pgvector extension
- Google Gemini embeddings
- psycopg2
- Python-dotenv

[Back to index](#contents)

## Project structure

```text
ai_faq_assistant/
├── app/
│   ├── __init__.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── corpus.py
│   │   ├── demo.py
│   │   ├── evaluation.py
│   │   ├── markdown.py
│   │   ├── strategies.py
│   │   └── types.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── domain/
│   │   ├── __init__.py
│   │   └── faq.py
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── embedding_client.py
│   │   └── faq_repository.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── ingestion.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retrieval.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── AI_assistant_service.py
│   │   ├── faq_ingestion_service.py
│   │   └── rag_service.py
├── data/
│   ├── faqs.json
│   └── policies/
│       ├── manifest.json
│       ├── remote_work_v1.md
│       ├── remote_work_v2.md
│       └── travel_expense_v1.md
├── sql/
│   ├── 001_enable_pgvector.sql
│   ├── 002_create_faq_table.sql
│   ├── 003_vector_search.sql
│   └── 004_insert_sample_faqs.sql
├── tests/
│   ├── test_chunking.py
│   └── test_core.py
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── main.py
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

[Back to index](#contents)

## Prerequisites

Before running the project, make sure you have:

1. Python 3.13 installed
2. PostgreSQL installed and running
3. pgvector enabled in your database
4. A Gemini API key

[Back to index](#contents)

## Local setup

### 1) Clone the repository

```bash
git clone https://github.com/nandinikori/FAQ_Assistant_Using_RAG.git
cd FAQ_Assistant_Using_RAG
```

### 2) Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, with uv:

```bash
uv sync
```

### 4) Configure environment variables

Create a `.env` file in the project root based on `.env.example`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_TEXT_MODEL=gemini-2.5-flash
DB_HOST=localhost
DB_NAME=ai_workshop
DB_USER=postgres
DB_PASSWORD=your_postgres_password
DB_PORT=5432
APP_ENV=development
```

### 5) Set up PostgreSQL and pgvector

Connect to your PostgreSQL database and enable the extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Then create the knowledge base if needed:

```sql
CREATE DATABASE ai_workshop;
```

[Back to index](#contents)

## Run the application

From the project root:

```bash
python main.py
```

The CLI menu supports:

1. Ingest FAQ data into PostgreSQL
2. Retrieve and display matching FAQs without AI generation
3. Ask Gemini to format an answer using retrieved FAQ context
4. Exit

To run the sample batch of FAQ questions and one open-ended question:

```bash
python -m app.services.AI_assistant_service
```

[Back to index](#contents)

## Data flow

```text
FAQ JSON file
    ↓
FAQIngestionService
    ↓
Gemini embedding generation
    ↓
PostgreSQL + pgvector storage
    ↓
Similarity search by embedding distance
    ↓
RAGService response generation
```

[Back to index](#contents)

## Testing

This project includes a minimal pytest suite to validate the main domain and service behavior.

```bash
pytest
```

The tests cover:

- settings validation
- FAQ record mapping
- JSON ingestion parsing
- repository insert logic
- service behavior for retrieval and answer assembly

[Back to index](#contents)

## Notes for public GitHub use

- Keep your `.env` file local and never commit secrets.
- Use `.env.example` as the template for contributors.
- Keep database credentials out of version control.
- This repository is structured as a clean demo/portfolio project rather than a throwaway notebook.

[Back to index](#contents)

## Development progress

This section records the repository's implementation as it evolves. Each entry describes capabilities present at that stage; planned capabilities are not counted as completed work.

<a id="day-1"></a>
### Day 1 — Basic FAQ RAG baseline

The starting implementation demonstrates the complete retrieval-augmented generation flow: load a knowledge source, retrieve relevant information by meaning, and provide that information to a language model to answer a question.

#### What is implemented

- **Ingestion:** load FAQ question–answer pairs from JSON and generate an embedding for each combined pair.
- **Storage:** persist FAQ content and embeddings in PostgreSQL with pgvector.
- **Retrieval:** embed the user's question and rank FAQ records using cosine distance. The retrieval interface defaults to five results; answer generation uses three.
- **Generation:** supply retrieved FAQ text to Gemini with instructions to answer from that information and avoid inventing facts.
- **Empty-context handling:** return a no-match message without generating an answer when no context is available.
- **Application structure:** separate configuration, domain models, database access, ingestion, retrieval, and application services.
- **Usage and testing:** provide an interactive CLI, a Dockerfile, setup instructions, and unit tests covering selected repository and service behavior with fake dependencies.

#### How the baseline works

```mermaid
flowchart TD
    A["FAQ JSON"] --> B["Embed question-answer pairs"]
    B --> C["PostgreSQL and pgvector"]
    D["User question"] --> E["Embed the question"]
    E --> F["Rank FAQ records by cosine distance"]
    C --> F
    F --> G["Assemble the top three FAQs"]
    G --> H["Generate an answer with Gemini"]
```

#### Concept at a glance

```mermaid
mindmap
  root((Day 1 RAG))
    Knowledge
      FAQ question-answer pairs
      JSON ingestion
    Embeddings
      Represent text numerically
      Compare semantic similarity
    Retrieval
      PostgreSQL and pgvector
      Nearest FAQ matches
    Generation
      Retrieved text as context
      Gemini answer
```

#### What this baseline establishes

A modular implementation of one dense-retrieval RAG pattern, from data ingestion to generated answers. Each FAQ pair acts as a natural retrieval unit; document chunking is not implemented at this stage.

#### Current limitations

- Retrieval returns the nearest records without a calibrated relevance or evidence-sufficiency check. Nonempty context can still be unrelated to the question.
- Repeated ingestion can create duplicate records: the method named `upsert` currently performs inserts.
- The SQL setup and application-created schema differ; the Docker Python version also differs from the project requirement.
- Hybrid retrieval, reranking, source citations, tenant authorization, and document version management are not implemented.
- There is no published retrieval/answer-quality benchmark, production monitoring, or load-test evidence.
- Existing unit tests do not establish real database integration correctness or end-to-end answer quality. This progress entry is based on source review; no application execution or benchmark was performed for this update.

[Back to index](#contents)

<a id="day-2-progress"></a>
### Day 2: Document Ingestion and Chunking

**Day 2 progress:** added a separate offline document pipeline to demonstrate and compare chunking strategies on versioned synthetic policy files. It preserves source locations and policy metadata, and includes an enterprise-style ingestion-to-retrieval architecture. The original FAQ example in `data/faqs.json` remains unchanged.

The corpus is under [`data/policies/`](./data/policies/) and includes headings, eligibility and reimbursement tables, exceptions, effective dates, and two versions of the remote-work policy.

Run the offline comparison from the project root:

```bash
python -m app.chunking.demo
```

It prints JSON metrics by document version and strategy: chunk count, average and maximum lexical-token count, target-size violations, and source-character coverage. These are structural diagnostics, **not** retrieval-quality or production-performance claims.

#### Chunking concepts and visuals

| Strategy | Boundary behavior | Best suited to |
| --- | --- | --- |
| Token window | Fixed-size windows with configurable overlap and character offsets from an injectable tokenizer. The default lexical tokenizer is deterministic but approximate; it is not a model-token billing count. | Baselines, unstructured text, and controlled size comparisons. |
| Recursive | Recursively prefers paragraph, newline, and whitespace boundaries before falling back to token-offset splits. | General text when fixed windows break useful boundaries. |
| Paragraph/sentence-aware | Packs paragraph and sentence units up to a target; splits very long sentences into token-like units. | Prose where readable boundaries matter. |
| Structure-aware | Preserves Markdown heading paths and logical blocks; keeps tables as row groups with repeated headers. | Policies, manuals, and structured knowledge. |
| Table-aware | Groups table rows and repeats the table header in every emitted row group. | Tabular rules, limits, and eligibility matrices. |
| Semantic | Splits adjacent blocks when an injected similarity score falls below a threshold, while respecting section and size boundaries. | Content with topic shifts that do not align with a fixed length. |
| Parent–child | Emits larger contextual parent chunks and smaller overlapping retrieval children linked by parent ID. | Precise retrieval with expansion to broader context. |

##### 1. Token-window chunking

```mermaid
flowchart LR
    T["Token stream: 1 2 3 4 5 6 7 8"] --> C1["Chunk 1: 1 2 3 4"]
    C1 -->|"overlap: 3 4"| C2["Chunk 2: 3 4 5 6"]
    C2 -->|"overlap: 5 6"| C3["Chunk 3: 5 6 7 8"]
```

##### 2. Recursive chunking

```mermaid
flowchart TD
    A["Whole document"] --> B{"Fits token limit?"}
    B -->|"Yes"| C["Emit chunk"]
    B -->|"No"| D{"Split at paragraph"}
    D --> E{"Each part fits?"}
    E -->|"Yes"| F["Emit paragraph chunks"]
    E -->|"No"| G{"Split at newline, then whitespace"}
    G --> H["Token-boundary fallback for oversized text"]
```

##### 3. Paragraph/sentence-aware chunking

```mermaid
flowchart LR
    A["Paragraph"] --> B["Sentence 1"]
    A --> C["Sentence 2"]
    A --> D["Sentence 3"]
    B --> E["Pack complete sentences up to size limit"]
    C --> E
    D --> E
    E --> F["Emit readable chunk(s)"]
```

##### 4. Structure-aware chunking

```mermaid
flowchart TD
    A["Policy document"] --> B["Heading: Eligibility"]
    B --> C["Clause and list blocks"]
    A --> D["Heading: Exceptions"]
    D --> E["Exception clauses"]
    C --> F["Chunks retain heading path"]
    E --> F
```

##### 5. Table-aware chunking

```mermaid
flowchart TD
    A["Table header + rows 1–N"] --> B["Partition rows by size"]
    B --> C["Chunk 1: header + rows 1–k"]
    B --> D["Chunk 2: repeated header + rows k+1–m"]
    B --> E["Chunk 3: repeated header + remaining rows"]
    C --> F["Each row group keeps source row locations"]
    D --> F
    E --> F
```

##### 6. Semantic chunking

```mermaid
flowchart LR
    A["Block 1"] --> S1["Similarity score"]
    B["Block 2"] --> S1
    S1 --> C{"Score below threshold?"}
    C -->|"No"| D["Keep blocks in same chunk"]
    C -->|"Yes"| E["Start a new chunk at topic shift"]
```

##### 7. Parent–child chunking

```mermaid
flowchart TD
    A["Source section"] --> B["Parent chunk: broad context"]
    B --> C["Child 1: precise retrieval"]
    B --> D["Child 2: precise retrieval"]
    B --> E["Child 3: precise retrieval"]
    C --> F["Retrieve child, follow parent ID"]
    D --> F
    E --> F
    F --> G["Supply parent context to answer generation"]
```

The semantic demo injects a transparent lexical-overlap proxy so it runs offline. A real system should inject cosine similarity from its selected embedding model and evaluate that boundary signal against labeled queries. The chunker interface is deliberately independent of any model provider.

This is a comparative showcase of common text chunking families, not a claim that every possible technique is implemented. Code-aware parsing, multimodal page/layout chunking, model-specific late chunking, and embedding-time contextualization are natural extensions when the corresponding source formats and model capabilities are selected.

#### Provenance and versioning

Every document carries a stable document ID, version, source name, and manifest metadata. Parsing yields typed Markdown blocks with one-based line ranges, zero-based Python character offsets, heading path, block kind, and table headers/rows. Chunks retain document version, source offsets and lines, source block IDs, heading path, strategy parameters, and (for child chunks) their parent ID. IDs are deterministic for a given source version, content, boundaries, strategy, and metadata. A new policy revision is ingested as a new version rather than overwriting history.

#### Enterprise architecture mapping

```mermaid
flowchart TB
    subgraph INGEST["1. Ingest and prepare"]
        direction LR
        SOURCE["Source registry<br/>Versioned documents"] --> ADAPTER["Ingestion<br/>adapter"]
        ADAPTER --> PARSE["Parse<br/>Normalize<br/>Classify"]
        PARSE --> BLOCKS["Canonical blocks<br/>+ provenance"]
    end

    subgraph INDEX["2. Chunk and index"]
        direction LR
        STRATEGY["Strategy registry<br/>+ chunking"] --> VALIDATE["Chunk validation<br/>+ evaluation"]
        VALIDATE --> EMBED["Embedding<br/>provider"]
        EMBED --> STORE["Versioned chunk store<br/>+ vector index"]
    end

    subgraph SERVE["3. Retrieve and answer"]
        direction LR
        SEARCH["Authorized<br/>filtered retrieval"] --> EXPAND["Parent expansion<br/>Rerank + citations"]
        EXPAND --> RESPONSE["RAG response"]
    end

    BLOCKS --> STRATEGY
    STORE --> SEARCH

    subgraph CONTROLS["Cross-cutting observability"]
        direction LR
        AUDIT["Audit events<br/>Retries"] --- METRICS["Metrics<br/>Evaluation"]
    end

    ADAPTER -.-> AUDIT
    VALIDATE -.-> METRICS
    SEARCH -.-> METRICS
```

The implementation in this repository covers the versioned corpus, Markdown parsing, chunking strategies, source lineage, deterministic IDs, and offline chunk-shape metrics. The remaining boxes describe enterprise integration points, not features claimed as already implemented. In production, source adapters publish versioned ingestion jobs; parsing and embedding workers are idempotent and retryable; chunk records retain source/version/strategy/model metadata; vector retrieval applies tenant and document-version authorization filters; child hits expand to parent context; and metrics, audit events, and human-reviewed relevance judgments feed evaluation. The legacy FAQ database schema and runtime are not yet aligned for a single production deployment, so this chunking demo intentionally runs without PostgreSQL or API credentials.

[Back to index](#contents)

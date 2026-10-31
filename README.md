# VectorDB

A vector database built from scratch in Python, with a web UI.

It implements **HNSW**, **KD-Tree** and **brute-force** nearest-neighbour search side by side, three distance metrics, and a **RAG pipeline** (retrieve, then generate) powered by local models through [Ollama](https://ollama.com).

> An educational project showing how systems like Pinecone, Weaviate, Chroma and Milvus work under the hood.

## Features

| Feature | Description |
|---|---|
| 3 search algorithms | HNSW (approximate, graph-based), KD-Tree (exact, low-dimensional), Brute Force (exact baseline) |
| 3 distance metrics | Cosine, Euclidean, Manhattan |
| Demo dataset | 20 hand-built 16-D vectors in 4 categories (CS, Math, Food, Sports) |
| PCA scatter plot | Live 2-D visualisation of the semantic space |
| Document embedding | Paste text, it is chunked and embedded with `nomic-embed-text` (768-D) |
| RAG | Ask questions, HNSW retrieves context, a local LLM answers |
| REST API | Insert, delete, search, benchmark, graph introspection |

## How it works

```
Your text
   │
   ▼
Ollama (nomic-embed-text)     text → 768-D vector
   │
   ▼
HNSW index (Python/NumPy)     multilayer proximity graph
   │
   ▼
Semantic search               nearest neighbours in vector space
   │
   ▼
Ollama (llama3.2)             reads the retrieved chunks, writes an answer
```

## Project structure

```
vectordb/
├── pyproject.toml
├── requirements.txt
├── src/vectordb/
│   ├── __main__.py            CLI entry point
│   ├── config.py              Settings (env-var overridable)
│   ├── core/
│   │   ├── metrics.py         cosine / euclidean / manhattan
│   │   └── models.py          VectorItem, Neighbor
│   ├── indexes/
│   │   ├── base.py            VectorIndex interface
│   │   ├── brute_force.py     exact linear scan
│   │   ├── kd_tree.py         exact, axis-aligned partitioning
│   │   └── hnsw.py            hierarchical navigable small world graph
│   ├── database/
│   │   ├── vector_db.py       16-D demo store, all three indexes in sync
│   │   └── document_db.py     HNSW store for real embeddings
│   ├── services/
│   │   ├── ollama_client.py   embeddings + generation
│   │   └── chunker.py         overlapping word-window chunking
│   ├── data/demo.py           bundled demo vectors
│   ├── api/                   Flask app factory + blueprints
│   │   ├── app.py  vectors.py  documents.py  system.py  errors.py
│   └── static/index.html      web UI
└── tests/                     unit and API tests
```

## Setup

Requires **Python 3.10+** and [Ollama](https://ollama.com) (only for the Documents and Ask AI tabs).

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/VectorDB.git
cd VectorDB
python -m venv .venv

# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
pip install -e .

# 2. Pull the models (about 2.3 GB total)
ollama pull nomic-embed-text
ollama pull llama3.2

# 3. Run
vectordb            # or: python -m vectordb
```

Open **http://localhost:8080**.

Expected output:

```
=== VectorDB Engine v1.0.0 ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW + KD-Tree + BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

The search tab works without Ollama.

## Configuration

Set environment variables before starting the server (PowerShell: `$env:VECTORDB_PORT = "9000"`).

| Variable | Default | Purpose |
|---|---|---|
| `VECTORDB_HOST` | `0.0.0.0` | Bind address |
| `VECTORDB_PORT` | `8080` | HTTP port |
| `VECTORDB_DEBUG` | `0` | Set `1` for debug logging |
| `VECTORDB_OLLAMA_HOST` | `127.0.0.1` | Ollama host |
| `VECTORDB_OLLAMA_PORT` | `11434` | Ollama port |
| `VECTORDB_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `VECTORDB_GEN_MODEL` | `llama3.2` | Generation model |

Running slowly on a laptop? Use the 1B model: `ollama pull llama3.2:1b`, then `VECTORDB_GEN_MODEL=llama3.2:1b`.

## Using the app

**Search tab:** type a concept (`binary tree`, `sushi`, `calculus`), pick an algorithm and metric, and search. *Compare all algos* times all three on the same query.

**Documents tab:** paste text and click *Embed & Insert*. Long text is split into overlapping 250-word chunks, each embedded separately.

**Ask AI tab:** your question is embedded, the 3 nearest chunks are retrieved with HNSW, and the LLM answers using them as context.

## REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | k-NN search (`algo`: `hnsw`, `kdtree`, `bruteforce`) |
| POST | `/insert` | Insert a demo vector `{"metadata","category","embedding"}` |
| DELETE | `/delete/<id>` | Delete a demo vector |
| GET | `/items` | List demo vectors |
| GET | `/benchmark?v=...&k=5&metric=cosine` | Time all three algorithms |
| GET | `/hnsw-info` | Graph structure and per-layer stats |
| GET | `/stats` | Database statistics |
| POST | `/doc/insert` | `{"title","text"}`: chunk, embed and store |
| GET | `/doc/list` | List stored chunks |
| DELETE | `/doc/delete/<id>` | Delete a chunk |
| POST | `/doc/search` | `{"question","k"}`: retrieval only |
| POST | `/doc/ask` | `{"question","k"}`: full RAG |
| GET | `/status` | Ollama status and model info |

Errors are returned as `{"error": "..."}` with an appropriate HTTP status (400 for bad input, 503 when Ollama is unavailable).

```bash
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"

curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

## Algorithms

**HNSW** assigns each node a random maximum layer from an exponential distribution. Layer 0 holds every node; each higher layer is exponentially sparser with longer-range links. Search descends greedily from the top layer to find an entry point, then runs a beam search of width `ef` on layer 0. Typical cost is about O(log N).

**KD-Tree** splits space along one axis per level, cycling through the dimensions, and prunes subtrees that cannot contain a closer point. It is exact for Euclidean distance but degrades towards a linear scan as dimensionality grows, because almost nothing can be pruned. Under cosine or Manhattan the pruning bound is no longer strictly valid, so results are approximate.

**Brute force** compares the query against every vector: O(N·d), always exact, and the baseline the other two are measured against.

## Development

```bash
pip install -e ".[dev]"
python -m pytest        # or: python -m unittest discover -s tests -t .
ruff check src tests
```

Both the API tests and the index tests run without Ollama; the API tests use a fake embedding client injected through `create_app(services=...)`.

## Notes on the Python port

- HNSW removal now re-selects the entry point from the highest remaining layer. The original C++ picked an arbitrary node.
- Document insertion embeds every chunk before storing any, so an Ollama failure mid-document cannot leave a partial insert.
- `/doc/ask` returns HTTP 503 with an error message when Ollama is down, instead of returning an error string as the answer.
- Search and benchmark now reject unknown `metric` / `algo` values with a 400 instead of silently falling back.
- Hand-rolled JSON parsing is replaced by Flask's JSON handling.
- The UI uses `window.location.origin`, so it works on any port.

## Limitations

This is an educational project. Data lives in memory only (nothing is persisted across restarts), the server is Flask's development server (use `waitress` or `gunicorn` for anything real), and pure-Python HNSW is far slower than the C++ original or production engines.

## License

MIT
=======
# VectorX
Python-based vector search engine with HNSW, KD-Tree, semantic search, and local LLM-powered RAG.

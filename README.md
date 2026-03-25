# Multi Agent Advanced RAG with LoRa fine tuning on local QWen
Author: Yihao Chen

End-to-end **retrieval-augmented generation** over the **2026 FIA Formula 1 regulations**: PDF ingestion, clause-aware chunking, **Pinecone** vector search with **BGE** embeddings, and a **LangGraph** workflow with **multiple LLM-backed agents**. Structured routing and validation use **OpenAI** models; the final answer can be produced by **OpenAI** or by a **locally hosted Qwen2.5** model with optional **LoRA** weights. A **Flask** web UI streams pipeline phases and tokens over **SSE**.

---

## What this project implements

- **Regulation corpus pipeline:** Download official PDFs → optional **PaddleOCR-VL** layout-aware OCR to JSON → **clause / article–aware chunking** → JSONL under `chunks/`.
- **Vector RAG:** **SentenceTransformers** (`BAAI/bge-large-en-v1.5`) embeds chunks; vectors are stored in **Pinecone** with metadata for citations and hydration from local JSONL when needed.
- **Multi-agent graph (LangGraph):** Separate nodes for conversation memory, query planning, hybrid retrieval, relevance judging, reference resolution, answer synthesis, and answer support checking—with **one conditional retry** when the checker is not satisfied.
- **Fine-tuned model integration:** Answer synthesis can use **PEFT LoRA adapters** on **Qwen2.5-1.5B-Instruct** (default adapter path `models/Qwen2.5-1.5B-lora`), satisfying a “LoRA in the loop” requirement alongside API-based agents.
- **Interfaces:** **CLI** (`src.advanced_rag.run`), **REST + SSE** (`frontend/app.py`), and optional notebooks for training and evaluation.

---

## Official data: 2026 FIA F1 regulations (PDFs)

Place downloaded PDFs under `data/` (see [Data acquisition](#1-data-acquisition)). Sources:

| Section | Topic | URL |
|--------|--------|-----|
| **A** | General provisions | [fia_2026_f1_regulations_-_section_a_general_provisions_-_iss_02_-_2026-02-27.pdf](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_a_general_provisions_-_iss_02_-_2026-02-27.pdf) |
| **B** | Sporting | [fia_2026_f1_regulations_-_section_b_sporting_-_iss_05_-_2026-02-27.pdf](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_05_-_2026-02-27.pdf) |
| **C** | Technical | [fia_2026_f1_regulations_-_section_c_technical_-_iss_16_-_2026-02-27.pdf](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_c_technical_-_iss_16_-_2026-02-27.pdf) |
| **D** | Financial — F1 teams | [fia_2026_f1_regulations_-_section_d_financial_-_f1_teams_-_iss_05_-_2026-02-27.pdf](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_d_financial_-_f1_teams_-_iss_05_-_2026-02-27.pdf) |
| **E** | Financial — power unit manufacturers | [fia_2026_f1_regulations_-_section_e_financial_regulations_-_power_unit_manufacturers_-_iss_03_-_2025-12-10_0.pdf](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_e_financial_regulations_-_power_unit_manufacturers_-_iss_03_-_2025-12-10_0.pdf) |
| **F** | Operational | [fia_2026_f1_regulations_-_section_f_operational_-_iss_06_-_2026-02-27.pdf](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_f_operational_-_iss_06_-_2026-02-27.pdf) |

---

## OCR and text manipulation

This repo treats the regulations as **layout-first documents**: OCR produces structured JSON per page, then dedicated code **cleans revision markup** and **re-segments** text into regulation-aware chunks with metadata for retrieval.

### Document OCR (PaddleOCR-VL)

- **`src/ocr/paddle_ocr.py`** runs the **PaddleOCR-VL** document parser on every `*.pdf` under `data/` (recursively). The model rasterizes PDF pages internally and returns a **per-page JSON** structure: each page holds a `parsing_res_list` of blocks with `block_label` (e.g. paragraph, title, table), `block_content`, **bounding boxes**, and **`block_order`** for reading order.
- Output is one JSON file per PDF, named like `{pdf_name}_by_PaddleOCR-VL.json`, written next to the source PDF (under `--out-dir`, default `data/`).

### Strikethrough and “deleted” regulation text (PyMuPDF)

FIA PDFs often show **struck or magenta revision lines** over fragments of text. A naive OCR block can mix **current** and **obsolete** wording, which is bad for RAG.

- **`src/ocr/remove_strike.py`** opens the **same PDF** with **PyMuPDF**, maps each OCR block’s bbox into PDF coordinates, and collects **horizontal strike-like segments** from vector drawings (filtered by **revision-style colors**).
- It **rebuilds `block_content`** from PDF **text spans** that fall inside the block region, **dropping spans** that overlap strike geometry (tunable thresholds: coverage, midline distance, length ratios, etc.).
- It writes a sibling file **`…_by_PaddleOCR-VL_no_strike.json`**. For batch chunking you can pass **`--prefer-no-strike`** to `run_chunking` so only the cleaned JSON is used when both exist.

```bash
.venv/bin/python -m src.ocr.remove_strike data   # or a single JSON path
```

### Parsing and clause-aware chunking

- **`src/regulations_chunking/parser.py`** loads Paddle JSON into typed **`Page` / `Block`** objects, normalizes fields, and **sorts blocks by `block_order`** (falling back to `block_id`) so downstream logic follows visual document order, not file key order.
- **`src/regulations_chunking/pipeline.py`** implements **FIA-style structure**:
  - Detects **`ARTICLE …`** and **`APPENDIX …`** headings and **groups blocks** into contiguous section spans.
  - **Skips table-of-contents** regions so lists of article titles are not embedded as body text.
  - Drops noisy layout roles via **`ChunkingConfig.ignore_labels`** (headers, footers, footnotes, page numbers, etc.).
  - Inside articles, splits on **clause patterns** (e.g. top-level `E1.2`, nested `E1.2.1`, lettered sub-items) and applies **size limits** (`max_chars`, `overlap_ratio`) so each **`Chunk`** has embeddable `text` plus **metadata** (article / clause / appendix ids, source file, page indices) for citations and hybrid lookup.
- **`src/regulations_chunking/run_chunking.py`** wires this up: discover `*_by_PaddleOCR-VL*.json` under `data/`, run `build_chunks`, emit **`chunks/**/*.chunks.jsonl`** rows shaped for embedding (`text` + `metadata`).

---

## Repository layout

Paths are relative to the repo root.

```text
.
├── data/                          # Your PDFs + PaddleOCR-VL JSON (gitignored except .gitkeep)
├── chunks/                        # Chunk JSONL outputs (gitignored except .gitkeep)
├── models/                        # Local HF exports, e.g. Qwen LoRA adapter (large; often gitignored)
├── frontend/
│   ├── app.py                     # Flask app: pages, POST /api/chat, POST /api/chat/stream (SSE)
│   ├── templates/index.html       # Chat UI
│   └── static/                    # CSS/JS for streaming UX and synthesizer toggle
├── src/
│   ├── ocr/
│   │   ├── paddle_ocr.py          # Batch PDF → *_by_PaddleOCR-VL.json (PaddleOCR-VL)
│   │   └── remove_strike.py       # PyMuPDF: rebuild OCR text minus revision strikethrough → *_no_strike.json
│   ├── regulations_chunking/
│   │   ├── parser.py              # Load OCR JSON pages
│   │   ├── model.py               # Chunking configuration types
│   │   ├── pipeline.py            # Clause-aware chunk building
│   │   ├── run_chunking.py        # CLI: OCR JSON → chunks/*.chunks.jsonl
│   │   ├── demo_chunking.py       # Small demos / experiments
│   │   └── embed_and_upsert.py    # Embed chunks + upsert to Pinecone
│   ├── rag/
│   │   ├── retriever.py           # Pinecone query + local chunk text hydration
│   │   ├── run_retrieval.py       # CLI: semantic search smoke tests
│   │   └── answer.py              # Simpler one-shot RAG QA (retrieve + single OpenAI call)
│   └── advanced_rag/
│       ├── graph.py               # LangGraph: agent nodes and edges
│       ├── nodes.py               # OpenAI JSON agents + synthesizer dispatch
│       ├── prompts.py             # System prompts per agent
│       ├── state.py               # Typed graph state
│       ├── memory.py              # Short session memory for follow-ups
│       ├── retrieval.py           # Hybrid retriever: vector + exact clause ID lookup
│       ├── local_qwen_synthesizer.py  # Shared Qwen (+ optional LoRA) for final answer
│       ├── service.py           # AdvancedRegulationAssistant facade
│       └── run.py                 # CLI entry for the full multi-agent pipeline
├── requirements.txt               # Python dependencies (see PyTorch / Paddle notes inside)
├── qwen2.5_lora_fine_tuned.ipynb # LoRA fine-tuning workflow (Qwen)
├── evaluation.ipynb               # Evaluation / analysis notebook
└── test_questions.md              # Example questions (optional)
```

---

## Prerequisites

- **Python 3.10+** recommended (matches project tooling).
- **OpenAI API key** for all structured agents (planner, judges, checker, etc.).
- **Pinecone** account and index (serverless or hosted URL) for vector storage.
- **GPU** strongly recommended for PaddleOCR-VL and for local Qwen inference; CPU works but is slow.
- **PyTorch:** Install a build that matches your CUDA version; see comments in `requirements.txt` and [PyTorch install](https://pytorch.org/get-started/locally/).
- **PaddleOCR-VL (optional):** Uncomment / install `paddlepaddle` / `paddlepaddle-gpu` and `paddleocr[doc-parser]` per `requirements.txt` and the [PaddleOCR-VL model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL).

---

## End-to-end: how to run everything

Run all commands from the **repository root** unless noted.

### 0. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Then install PyTorch for your platform (see requirements.txt).
```

Create a **`.env`** file in the repo root (never commit it). Minimum:

```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_HOST=https://your-index-xxxx.svc.region.pinecone.io
# Optional overrides:
# PINECONE_INDEX_NAME=your-index-name
# PINECONE_NAMESPACE=regulations-2026
```

### 1. Data acquisition

1. Create `data/` if needed (a `data/.gitkeep` may already exist).
2. Download the six PDFs from the [table above](#official-data-2026-fia-f1-regulations-pdfs) into `data/` (any subdirectory is fine).

### 2. OCR (PDF → JSON)

Uses PaddleOCR-VL when installed. **Use the same interpreter** that has `paddleocr` installed. For what each stage does, see [OCR and text manipulation](#ocr-and-text-manipulation).

```bash
.venv/bin/python -m src.ocr.paddle_ocr --data-dir data --out-dir data
```

This writes one `*_by_PaddleOCR-VL.json` per PDF next to or under `data/`, preserving relative paths.

**Optional — remove struck revision text** (recommended for regulation PDFs):

```bash
.venv/bin/python -m src.ocr.remove_strike data
```

### 3. Chunking (JSON → `chunks/*.chunks.jsonl`)

Uses the pipeline described under [Parsing and clause-aware chunking](#parsing-and-clause-aware-chunking).

```bash
.venv/bin/python -m src.regulations_chunking.run_chunking \
  --input-dir data \
  --output-dir chunks
```

Use `--prefer-no-strike` if you maintain both base and `_no_strike` OCR JSON variants and want to prefer the latter.

### 4. Embed and upsert to Pinecone

Dry run (embed only, no upsert):

```bash
.venv/bin/python -m src.regulations_chunking.embed_and_upsert \
  --input-dir chunks \
  --create-index-if-missing \
  --dry-run
```

Full upsert (example namespace):

```bash
.venv/bin/python -m src.regulations_chunking.embed_and_upsert \
  --input-dir chunks \
  --create-index-if-missing \
  --namespace regulations-2026
```

Useful flags (see `embed_and_upsert.py --help`): `--index-name`, `--host`, `--limit`, batch sizes, `--store-text-in-metadata`, etc.

### 5. Sanity-check retrieval

```bash
.venv/bin/python -m src.rag.run_retrieval \
  --query "What are F1 team shutdown periods?" \
  --top-k 5
```

Pass `--index-name`, `--namespace`, and `--host` if they are not set in `.env`.  
If vectors omit inline text, the retriever **hydrates** from local `chunks/` using metadata.

### 6. Multi-agent RAG (CLI)

```bash
.venv/bin/python -m src.advanced_rag.run \
  --query "Explain power unit cost cap reporting." \
  --answer-synthesizer openai
```

**Local Qwen + LoRA** (adapter default: `models/Qwen2.5-1.5B-lora`):

```bash
.venv/bin/python -m src.advanced_rag.run \
  --query "..." \
  --answer-synthesizer local_qwen \
  --qwen-adapter-path models/Qwen2.5-1.5B-lora
```

**Base Qwen only** (no LoRA):

```bash
.venv/bin/python -m src.advanced_rag.run \
  --query "..." \
  --answer-synthesizer local_qwen \
  --qwen-base-only
```

### 7. Frontend (Flask)

```bash
source .venv/bin/activate
python -m frontend.app
```

Open **http://127.0.0.1:5000**. The UI can switch the **answer backend** (OpenAI vs Qwen vs base Qwen).

**JSON (no stream):** `POST /api/chat` with body `{"message": "...", "session_id": "optional", "synthesizer": "openai" | "local_qwen" | "local_qwen_base"}`.

**SSE stream:** `POST /api/chat/stream` with the same JSON; events include `phase`, `token`, and `meta`.

**Server-side defaults** (optional env):

- `ADVANCED_RAG_SYNTHESIZER=openai` | `local_qwen` | `local_qwen_base`
- `ADVANCED_RAG_QWEN_ADAPTER_PATH=/path/to/adapter`
- `ADVANCED_RAG_QWEN_BASE_ONLY=1`
- `ADVANCED_RAG_QWEN_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct`
- `ADVANCED_RAG_DEBUG=1` — logs Qwen load/generate timings

### 8. Simpler baseline RAG (optional)

Single retrieval + one OpenAI completion:

```bash
.venv/bin/python -m src.rag.answer --query "Your question"
```

---

## Agent workflow (high level)

1. **Conversation memory** — Resolves follow-ups using short history and active topics/clauses.  
2. **Query planner** — Produces search intent and structured hints for retrieval.  
3. **Retriever** — Hybrid **Pinecone** search plus optional **exact clause / article** fetches from `chunks/`.  
4. **Relevance judge** — Filters or adjusts what enters synthesis.  
5. **Reference resolver** — Aligns citations and regulation references with retrieved evidence.  
6. **Answer synthesizer** — **OpenAI** or **local Qwen (+ LoRA)** produces the user-facing answer.  
7. **Answer check** — Verifies support against sources; on failure, the graph may **retry retrieval + synthesis** once.

---

## Troubleshooting

- **`No module named 'paddleocr'`** — You are not using the venv where Paddle is installed; call `.venv/bin/python -m src.ocr.paddle_ocr`.
- **First Qwen run is slow** — Model download to `HF_HOME` and GPU warmup; reuse the process so `get_shared_qwen_synthesizer` does not reload weights every request.
- **Pinecone / hydration** — Keep `chunks/` on disk aligned with the corpus you indexed so metadata-only vectors still resolve to full text.

---

## References

- LangChain Hugging Face chat models: [Hugging Face integration](https://python.langchain.com/docs/integrations/chat/huggingface/)
- Pinecone: [Python client](https://docs.pinecone.io/)
- PaddleOCR-VL: [Hugging Face model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

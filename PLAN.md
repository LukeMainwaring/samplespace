# SampleSpace — Modernization Plan

## Context

The current Music Sample Assistant is a 3+ year old project (React 17, Flask, TensorFlow 2.7, Docker/K8s) with a single resume bullet. As a founding engineer targeting senior AI/ML roles at early-stage startups, it needs to reflect your current skill level. The goal is a **fresh repo** that serves as an impressive resume showcase demonstrating modern full-stack AI engineering — not a production product.

**Time budget:** ~25-40 hours over a week.

## What Makes This Impressive (2-3 min review)

1. **Multi-modal AI architecture** — Custom PyTorch CNN + CLAP audio embeddings + LLM agent orchestrating both. This is not a tutorial project.
2. **Stack consistency** — FastAPI + Next.js + Pydantic AI + PyTorch + pgvector mirrors the Neurocache project, showing depth rather than tech-hopping.
3. **Clean engineering** — Typed APIs, structured monorepo, tests, CI, Docker Compose.
4. **Great README** — Architecture diagram, demo GIF, "How it works" section. This is what reviewers actually read.

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Frontend | Next.js 15 (App Router) + Tailwind + shadcn/ui | Matches Neurocache. Modern, fast to build. |
| Backend | FastAPI + Pydantic v2 | Type-safe, fast, auto-generates OpenAPI. |
| Agent/LLM | Pydantic AI | Already on resume. Natural fit with FastAPI. |
| ML Model | PyTorch + torchaudio | Resume alignment. torchaudio replaces Kapre natively. |
| Embeddings | CLAP (`laion/larger_clap_music`) | "CLIP for audio" — enables NL queries over audio. Most impressive tech choice. |
| Database | PostgreSQL + pgvector | One DB for structured data + vector search. Same as Neurocache. |
| Audio analysis | librosa + music21 | Key detection, BPM, duration extraction. |
| DevOps | Docker Compose + GitHub Actions | Right level for a portfolio project. No K8s. |
| Testing | pytest (backend), Vitest (frontend) | Standard choices, selective coverage. |

**Dropped from v1:** Redux, MUI, Kubernetes, MySQL, TensorFlow, Kapre, separate ML service.

## Architecture: Why CLAP + CNN + Agent?

- **CLAP** (pretrained): Semantic search from natural language — "find me a warm pad." Bridges human description to audio content. Zero training required.
- **CNN** (custom-trained): Audio-to-audio similarity from learned spectral features. Shows custom ML engineering, not just API calls.
- **Pydantic AI Agent**: Orchestrates both modalities + metadata filtering. A query like "find a lead that goes well with this bass" triggers CNN similarity, key compatibility filtering, then CLAP ranking. This multi-tool orchestration is the agentic AI signal.

This layered design mirrors real AI product architecture — knowing when to use which modality is the skill.

## Project Structure

```
samplespace/
├── README.md                          # Architecture diagram, demo GIF, setup
├── docker-compose.yml                 # Full stack: frontend, backend, db
├── .github/workflows/ci.yml
├── frontend/
│   ├── src/app/                       # Next.js App Router
│   │   ├── page.tsx                   # Main sample browser
│   │   └── layout.tsx
│   ├── src/components/
│   │   ├── sample-browser.tsx
│   │   ├── sample-card.tsx
│   │   ├── audio-player.tsx
│   │   ├── chat-panel.tsx             # Agent chat with tool-call transparency
│   │   └── waveform-viz.tsx
│   └── Dockerfile
├── backend/
│   ├── pyproject.toml                 # uv for deps
│   ├── src/samplespace/
│   │   ├── main.py                    # FastAPI app
│   │   ├── config.py                  # Pydantic Settings
│   │   ├── api/
│   │   │   ├── samples.py             # CRUD + search endpoints
│   │   │   └── agent.py               # Streaming chat endpoint
│   │   ├── services/
│   │   │   ├── audio_analysis.py      # librosa key/BPM extraction
│   │   │   ├── embedding_service.py   # CLAP embedding generation
│   │   │   └── agent_service.py       # Pydantic AI agent + tools
│   │   ├── ml/
│   │   │   ├── model.py               # PyTorch CNN (dual-head: classify + embed)
│   │   │   ├── dataset.py             # torchaudio Dataset
│   │   │   ├── train.py               # Training script
│   │   │   └── predict.py             # Inference wrapper
│   │   ├── models/
│   │   │   ├── db.py                  # SQLAlchemy models
│   │   │   └── schemas.py             # Pydantic schemas
│   │   └── db/session.py              # Async SQLAlchemy + Alembic
│   ├── tests/
│   └── Dockerfile
├── data/samples/                      # Audio files (gitignored, seeded via script)
└── scripts/
    ├── seed.py                        # Download/populate sample data
    └── embed_samples.py               # Batch CLAP embedding
```

**Key decision:** One backend service, not three. The ML model loads in-process — a separate service adds complexity without demonstrating anything at this scale.

## Implementation Phases

### Phase 0: Repo Setup
- Create fresh repo with `PLAN.md`, `CLAUDE.md`, `.gitignore`
- Initial commit, create GitHub repo, push

### Phase 1: Foundation (~6 hrs)
- Monorepo directory structure
- FastAPI scaffold: app factory, Pydantic Settings, async SQLAlchemy + Postgres
- Alembic migration for `samples` table (id, filename, key, bpm, duration, type, clap_embedding vector(512), cnn_embedding vector(128), created_at)
- Next.js scaffold: App Router, Tailwind, shadcn/ui, basic layout
- Docker Compose: Postgres (pgvector), backend, frontend
- Seed script: 50-100 CC-licensed samples from Freesound
- Audio analysis service: librosa key/BPM/duration extraction on ingestion

**Checkpoint:** `docker compose up` runs. Seed populates DB. `GET /samples` returns metadata.

### Phase 2: CLAP Embeddings + Semantic Search (~5 hrs)
- Load `laion/larger_clap_music` from HuggingFace transformers
- `embed_audio(path) -> vector` and `embed_text(query) -> vector` functions
- Batch embedding script for seeded samples
- `POST /samples/search` — encode query with CLAP text encoder, pgvector cosine similarity, metadata filters
- Frontend search bar with results grid + audio players

**Checkpoint:** Type "warm analog pad" and get relevant results.

### Phase 3: PyTorch CNN (~6 hrs)
- torchaudio Dataset class: load audio, compute mel spectrograms
- CNN architecture: 4-5 conv blocks, global avg pool, **dual-head** (classification logit + 128-dim embedding)
- Training script with data augmentation (time stretch, pitch shift via torchaudio)
- Inference wrapper: load checkpoint, return embedding + predicted category
- `GET /samples/{id}/similar` — CNN embedding nearest neighbors

**Checkpoint:** CNN trains. Similarity endpoint returns results. (Small dataset will overfit — pipeline is the point. README acknowledges this and mentions NSynth as scaling path.)

### Phase 4: Pydantic AI Agent + Chat UI (~6 hrs)
- Agent with tools:
  - `search_by_description()` — CLAP semantic search
  - `find_similar_samples()` — CNN similarity
  - `check_key_compatibility()` — circle of fifths / music theory
  - `analyze_sample()` — return full metadata
  - `suggest_complement()` — combines CNN + key/BPM filtering
- System prompt with music production context
- `POST /agent/chat` — SSE streaming with tool-call metadata
- Frontend chat panel: Vercel AI SDK `useChat`, inline tool-call display

**Checkpoint:** Chat like "Find a lead that goes with this bass sample" triggers multi-step tool orchestration.

### Phase 5: Polish + README (~7 hrs)
- Waveform visualization (wavesurfer.js)
- Sample detail view: metadata, spectrogram, similar samples
- README: architecture diagram (Mermaid), demo GIF, "How it works", tech stack, setup, design decisions
- Selective tests (3-5): audio analysis, embedding shape, agent tool routing, search endpoint
- Type hints, docstrings on public functions, consistent error handling
- GitHub Actions CI: ruff lint, pyright, pytest

**Checkpoint:** Portfolio-ready. Clean README tells the whole story.

## Resume Bullet Update (after completion)

Current:
> Trained a CNN on music sample spectrograms to suggest complementary sounds during production.

Suggested (2 bullets):
> Built a multi-modal music sample assistant combining a custom PyTorch CNN for spectrogram similarity, CLAP embeddings for natural language audio search, and a Pydantic AI agent orchestrating both to answer queries like "find a warm pad in D minor at 120 BPM."

> Designed the audio pipeline with torchaudio for spectrogram generation, librosa for key/BPM detection, and pgvector for semantic similarity search across CLAP embeddings.

## Verification

- `docker compose up` starts everything (Postgres, backend, frontend)
- `python scripts/seed.py` populates the database with samples
- `python scripts/embed_samples.py` generates CLAP embeddings
- Browse samples at `localhost:3000`, play audio, filter by key/BPM
- Search "bright synth lead" and get semantically relevant results
- Click a sample, see CNN-recommended similar samples
- Open chat, ask "find me something that complements this bass line in A minor" — agent calls multiple tools and returns ranked results
- `pytest` passes, `ruff check` clean, `pyright` clean

## Risks

| Risk | Mitigation |
|------|------------|
| CLAP model ~600MB, slow cold start | Load once at startup. Mock in tests/CI. |
| Small training set for CNN | Acknowledge in README. Pipeline > results. Mention NSynth as scaling path. |
| Pydantic AI + Vercel AI SDK streaming | Reuse pattern from Neurocache. |
| Scope creep | Core loop: ingest -> embed -> search -> agent. Cut everything else if behind. |

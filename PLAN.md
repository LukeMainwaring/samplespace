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
| Frontend | Next.js 16 (App Router) + Tailwind + shadcn/ui | Matches Neurocache. Modern, fast to build. |
| Frontend linting | Biome via Ultracite | Matches Neurocache. Fast, consistent. |
| Frontend data | TanStack Query + OpenAPI client gen | Type-safe generated client from backend OpenAPI. Same pattern as Neurocache. |
| Backend | FastAPI + Pydantic v2 (Python 3.13) | Type-safe, fast, auto-generates OpenAPI. |
| Agent/LLM | Pydantic AI with OpenAI | Already on resume. Natural fit with FastAPI. Same provider as Neurocache. |
| ML Model | PyTorch + torchaudio | Resume alignment. torchaudio replaces Kapre natively. |
| Embeddings | CLAP (`laion/larger_clap_music`) | "CLIP for audio" — enables NL queries over audio. Most impressive tech choice. |
| Database | PostgreSQL + pgvector | One DB for structured data + vector search. Same as Neurocache. |
| Audio analysis | librosa + music21 | Key detection, BPM, duration extraction. |
| Package management | uv (backend), pnpm (frontend) | Matches Neurocache. Modern, fast. |
| DevOps | Docker Compose + GitHub Actions | Right level for a portfolio project. No K8s. |
| Code quality | pre-commit (Ruff + mypy strict) | Matches Neurocache. Automated on every commit. |
| Testing | pytest (backend), Vitest (frontend) | Standard choices, selective coverage. |

**Dropped from v1:** Redux, MUI, Kubernetes, MySQL, TensorFlow, Kapre, separate ML service, Auth0.

## Architecture: Why CLAP + CNN + Agent?

- **CLAP** (pretrained): Semantic search from natural language — "find me a warm pad." Bridges human description to audio content. Zero training required.
- **CNN** (custom-trained): Audio-to-audio similarity from learned spectral features. Shows custom ML engineering, not just API calls.
- **Pydantic AI Agent**: Orchestrates both modalities + metadata filtering. A query like "find a lead that goes well with this bass" triggers CNN similarity, key compatibility filtering, then CLAP ranking. This multi-tool orchestration is the agentic AI signal.

This layered design mirrors real AI product architecture — knowing when to use which modality is the skill.

## Project Structure

```
samplespace/
├── .claude/
│   ├── settings.json
│   ├── rules/backend/code-conventions.md
│   ├── agents/
│   │   ├── code-reviewer.md
│   │   └── product-advisor.md
│   └── skills/
│       ├── create-pr/SKILL.md
│       ├── updating-deps/SKILL.md
│       └── explain-code/SKILL.md
├── .github/workflows/ci.yml
├── backend/
│   ├── pyproject.toml                     # uv for deps, Python 3.13
│   ├── Dockerfile                         # Multi-stage (local/hosted)
│   ├── scripts/
│   │   ├── create-db-revision-docker.sh
│   │   ├── migrate-docker.sh
│   │   ├── downgrade-db-revision-docker.sh
│   │   ├── run_local.sh
│   │   └── run_hosted.sh
│   ├── src/samplespace/
│   │   ├── app.py                         # FastAPI app factory + lifespan
│   │   ├── core/config.py                 # Pydantic Settings
│   │   ├── routers/
│   │   │   ├── main.py                    # Router registry
│   │   │   ├── samples.py                 # CRUD + search endpoints
│   │   │   ├── agent.py                   # Streaming chat endpoint
│   │   │   └── health.py
│   │   ├── agents/
│   │   │   ├── sample_agent.py            # Pydantic AI agent + system prompt
│   │   │   ├── deps.py                    # AgentDeps dataclass
│   │   │   └── tools/
│   │   │       ├── clap_tools.py          # CLAP semantic search
│   │   │       ├── cnn_tools.py           # CNN similarity
│   │   │       └── analysis_tools.py      # Key compat, metadata
│   │   ├── models/                        # SQLAlchemy (re-exports in __init__)
│   │   │   ├── base.py                    # CRUD base class
│   │   │   └── sample.py                  # Sample with pgvector columns
│   │   ├── schemas/
│   │   │   └── sample.py                  # Pydantic request/response schemas
│   │   ├── services/
│   │   │   ├── audio_analysis.py          # librosa key/BPM extraction
│   │   │   ├── embedding.py               # CLAP embedding generation
│   │   │   └── sample.py                  # Sample business logic
│   │   ├── ml/
│   │   │   ├── model.py                   # PyTorch CNN (dual-head)
│   │   │   ├── dataset.py                 # torchaudio Dataset
│   │   │   ├── train.py                   # Training script
│   │   │   └── predict.py                 # Inference wrapper
│   │   ├── dependencies/                  # FastAPI DI (db, OpenAI client)
│   │   ├── migrations/                    # Alembic
│   │   └── utils/
│   └── tests/
├── frontend/
│   ├── package.json                       # pnpm
│   ├── Dockerfile
│   ├── next.config.ts
│   ├── tsconfig.json
│   ├── biome.jsonc                        # Biome via Ultracite
│   ├── components.json                    # shadcn/ui
│   ├── openapi-ts.config.ts               # OpenAPI client gen
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                       # Main sample browser
│   │   └── api/chat/route.ts              # Proxy to backend agent
│   ├── api/
│   │   ├── client.ts                      # Axios config
│   │   ├── hooks/                         # Custom TanStack Query hooks
│   │   └── generated/                     # Auto-generated (do not edit)
│   ├── components/
│   │   ├── chat-panel.tsx                 # Agent chat with tool-call transparency
│   │   ├── sample-browser.tsx
│   │   ├── sample-card.tsx
│   │   ├── audio-player.tsx
│   │   ├── waveform-viz.tsx
│   │   └── ui/                            # shadcn/ui
│   ├── hooks/
│   └── lib/
├── docs/
│   ├── pydantic-ai-llms-full.txt          # Downloaded Pydantic AI reference
│   └── vercel-ai-sdk-ui.txt               # Downloaded Vercel AI SDK UI reference
├── data/samples/                           # Audio files (gitignored)
├── scripts/
│   ├── seed.py                            # Download/populate sample data
│   └── embed_samples.py                   # Batch CLAP embedding
├── .editorconfig
├── .gitignore
├── .pre-commit-config.yaml
├── .dockerignore
├── .env.sample
├── docker-compose.yml
├── CLAUDE.md
├── DEVELOPMENT.md
├── PLAN.md
└── README.md
```

**Key decision:** One backend service, not three. The ML model loads in-process — a separate service adds complexity without demonstrating anything at this scale.

## Implementation Phases

### Phase 0: Repo Setup & Scaffolding
- Create fresh repo with `PLAN.md`, `CLAUDE.md`, `.gitignore`
- Set up `.claude/` directory (settings.json, rules, agents, skills)
- Create root config files (`.editorconfig`, `.pre-commit-config.yaml`, `.dockerignore`, `.env.sample`)
- Create `DEVELOPMENT.md`
- Set up GitHub Actions CI (`.github/workflows/ci.yml`)
- Initial commit, create GitHub repo, push

**Checkpoint:** All scaffolding and configuration in place. Claude Code agents/skills work.

### Phase 1: Foundation (~6 hrs)
- Monorepo directory structure (`backend/`, `frontend/`, `docs/`, `scripts/`, `data/`)
- Backend: FastAPI scaffold with app factory + lifespan, Pydantic Settings (`core/config.py`), async SQLAlchemy + Postgres, `dependencies/` for DI
- Alembic migration for `samples` table (id, filename, key, bpm, duration, type, clap_embedding vector(512), cnn_embedding vector(128), created_at)
- Frontend: Next.js 16 scaffold with App Router, Tailwind, shadcn/ui, Biome/Ultracite, basic layout
- Frontend: OpenAPI client gen setup (`openapi-ts.config.ts`, `api/client.ts`, TanStack Query provider)
- Docker Compose: Postgres (pgvector), backend, frontend
- `docs/` directory with downloaded Pydantic AI and Vercel AI SDK UI docs
- Backend `pyproject.toml` with uv, Python 3.13, dev groups (ruff, mypy, pytest, pre-commit)
- Install pre-commit hooks (`uv run pre-commit install`)
- Seed script: 50-100 CC-licensed samples from Freesound
- Audio analysis service: librosa key/BPM/duration extraction on ingestion

**Checkpoint:** `docker compose up` runs. Seed populates DB. `GET /samples` returns metadata. Pre-commit hooks pass. CI runs on push.

### Phase 2: CLAP Embeddings + Semantic Search (~5 hrs)
- Load `laion/larger_clap_music` from HuggingFace transformers via lifespan
- `embed_audio(path) -> vector` and `embed_text(query) -> vector` functions in `services/embedding.py`
- Batch embedding script for seeded samples
- `POST /samples/search` — encode query with CLAP text encoder, pgvector cosine similarity, metadata filters
- Frontend search bar with results grid + audio players
- Regenerate frontend API client after adding search endpoint

**Checkpoint:** Type "warm analog pad" and get relevant results.

### Phase 3: PyTorch CNN (~6 hrs)
- torchaudio Dataset class: load audio, compute mel spectrograms
- CNN architecture: 4-5 conv blocks, global avg pool, **dual-head** (classification logit + 128-dim embedding)
- Training script with data augmentation (time stretch, pitch shift via torchaudio)
- Inference wrapper: load checkpoint, return embedding + predicted category
- `GET /samples/{id}/similar` — CNN embedding nearest neighbors
- Regenerate frontend API client after adding similarity endpoint

**Checkpoint:** CNN trains. Similarity endpoint returns results. (Small dataset will overfit — pipeline is the point. README acknowledges this and mentions NSynth as scaling path.)

### Phase 4: Pydantic AI Agent + Chat UI (~6 hrs)
- Agent in `agents/sample_agent.py` with tools registered via `register_*_tools()`:
  - `search_by_description()` — CLAP semantic search
  - `find_similar_samples()` — CNN similarity
  - `check_key_compatibility()` — circle of fifths / music theory
  - `analyze_sample()` — return full metadata
  - `suggest_complement()` — combines CNN + key/BPM filtering
- System prompt with music production context
- `POST /agent/chat` — SSE streaming with tool-call metadata (Vercel AI SDK format)
- Frontend chat panel: Vercel AI SDK `useChat`, inline tool-call display
- `AgentDeps` dataclass for dependency injection into tools

**Checkpoint:** Chat like "Find a lead that goes with this bass sample" triggers multi-step tool orchestration.

### Phase 5: Polish + README (~7 hrs)
- Waveform visualization (wavesurfer.js)
- Sample detail view: metadata, spectrogram, similar samples
- README: architecture diagram (Mermaid), demo GIF, "How it works", tech stack, setup, design decisions
- Selective tests (3-5): audio analysis, embedding shape, agent tool routing, search endpoint
- Type hints, docstrings on public functions, consistent error handling

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
- Browse samples at `localhost:3002`, play audio, filter by key/BPM
- Search "bright synth lead" and get semantically relevant results
- Click a sample, see CNN-recommended similar samples
- Open chat, ask "find me something that complements this bass line in A minor" — agent calls multiple tools and returns ranked results
- `pytest` passes, `ruff check` clean, `mypy --strict` clean
- `uv run pre-commit run --all-files` passes
- `pnpm lint` passes

## Risks

| Risk | Mitigation |
|------|------------|
| CLAP model ~600MB, slow cold start | Load once at startup via lifespan. Mock in tests/CI. |
| Small training set for CNN | Acknowledge in README. Pipeline > results. Mention NSynth as scaling path. |
| Pydantic AI + Vercel AI SDK streaming | Reuse pattern from Neurocache. |
| Scope creep | Core loop: ingest -> embed -> search -> agent. Cut everything else if behind. |

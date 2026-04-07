# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See @README.md for a project overview and @DEVELOPMENT.md for a development guide.

When working with this codebase, prioritize readability over cleverness. Ask clarifying questions before making architectural changes.

## Common Commands

### Backend (Python)

```bash
# Install dependencies
uv sync --directory backend

# Run backend with Docker (includes PostgreSQL)
docker compose up -d

# Pre-commit hooks (covers type checking, linting, and formatting for backend)
uv run --directory backend pre-commit run --all-files

# Create local database migration
./backend/scripts/create-db-revision-docker.sh "<migration_message>"

# Apply pending migrations (ask user first)
./backend/scripts/migrate-docker.sh
```

### Frontend (TypeScript/Next.js)

```bash
pnpm -C frontend install
pnpm -C frontend lint            # lint with ultracite
pnpm -C frontend format          # format with ultracite
pnpm -C frontend generate-client # regenerate API client from backend OpenAPI
```

After making frontend code changes, run `pnpm -C frontend format` to fix formatting. Use `pnpm -C frontend lint` to check for errors. Do not run `pnpm -C frontend build` for validation -- it's slow and rarely catches issues that linting misses. The dev server (`pnpm -C frontend dev`) is typically already running during development.

## Architecture

### Backend (`backend/`)

FastAPI async backend. Layered as: `routers/` (thin HTTP handlers) → `services/` (business logic) → `models/` (SQLAlchemy with CRUD classmethods).

-   **`src/samplespace/app.py`**: Entry point with CORS middleware and lifespan handler (CLAP + CNN model loading)
-   **`src/samplespace/routers/`**: API routes by domain (samples, agent, health)
-   **`src/samplespace/agents/`**: Pydantic AI agent definition and tooling
    -   `sample_agent.py` — agent definition, system prompt, capability registration
    -   `deps.py` — shared `AgentDeps` (db, CLAP, CNN, thread_id, song_context)
    -   `capabilities/` — modular feature sets composed into the agent: search (CLAP/CNN/upload), analysis (key compatibility, metadata), context (song context management + instruction injection), pairing (pair presentation, verdicts, preference learning), production (kit building, transforms, previews)
    -   `tools/` — individual tool implementations grouped by domain (one file per capability area)
-   **`src/samplespace/services/`**: Business logic grouped by domain — audio pipeline (analysis, embedding, spectrogram, transforms), search/retrieval (sample CRUD, candidate search), pairing (pair scoring, pair features, preferences), production (kit building, kit preview mixing), and upload processing
-   **`src/samplespace/models/`**: SQLAlchemy async models (Sample with pgvector embedding columns, PairVerdict, Thread)
-   **`src/samplespace/ml/`**: Custom dual-head CNN — model architecture, torchaudio dataset with augmentation, training loop (SupCon + CE loss, mixup, cosine annealing), inference wrapper
-   **`src/samplespace/schemas/`**: Pydantic schemas for API contracts
-   **`src/samplespace/core/config.py`**: Settings via pydantic-settings (reads from `.env`)
-   **`src/samplespace/dependencies/`**: FastAPI dependency injection (db sessions, OpenAI client, CLAP models, CNN model)
-   **`src/samplespace/migrations/`**: Alembic migrations for PostgreSQL + pgvector

See `.claude/rules/backend/code-conventions.md` for code style and conventions.

### Frontend (`frontend/`)

Next.js 16 with App Router.

-   **`app/page.tsx`**: Main sample browser page
-   **`app/api/chat/route.ts`**: Proxy route that forwards chat requests to backend agent
-   **`components/chat.tsx`**: Chat orchestrator using `@ai-sdk/react` useChat hook; fetches and passes song context to header; manages file attachment state for uploads; wraps messages with `ChatActionsProvider` for verdict buttons
-   **`components/messages.tsx`**: Message list container with auto-scroll
-   **`components/message.tsx`**: Individual message rendering and loading state
-   **`components/multimodal-input.tsx`**: Chat input with file attachment and local storage persistence
-   **`components/song-context-badge.tsx`**: Read-only badge displaying active song context (key/BPM/genre/vibe)
-   **`components/sample-browser.tsx`**: Sample grid with key/BPM/type filters; split-pane layout driven by `selectedSampleId` — when a sample is selected, the list compresses to the left and a detail panel appears on the right
-   **`components/sample-detail-panel.tsx`**: Splice-style inline detail panel showing full metadata, waveform, mel spectrogram (full/CNN toggle), and CNN-similar samples with similarity percentages; manages its own playback state independently from the sample list
-   **`components/candidate-samples.tsx`**: Upload panel for reference tracks with playback, metadata editing, and delete functionality
-   **`components/sample-metadata-dialog.tsx`**: Post-upload dialog for correcting auto-detected key, BPM, and loop/one-shot classification
-   **`components/elements/sample-card.tsx`**: Shared sample card component used by pair-verdict-block, kit-block, and sample-results-block
-   **`components/elements/sample-results-block.tsx`**: Renders `sample-results` code fences as a vertical list of playable SampleCards (used by all search/similarity tools)
-   **`components/audio-player.tsx`**: Audio playback controls
-   **`components/waveform-viz.tsx`**: wavesurfer.js waveform rendering
-   **`api/client.ts`**: Generated client configuration (baseURL from `lib/constants.ts`, credentials)
-   **`api/hooks/`**: Custom TanStack Query hooks wrapping generated client
-   **`api/generated/`**: Auto-generated TypeScript client from OpenAPI (do not edit manually)
-   **`components/ui/`**: Reusable UI components (shadcn/ui style)

### Data Flow

1. Frontend `useChat` → `/api/chat` route → proxied to backend `POST /agent/chat`
2. Backend loads thread's `song_context` and injects into `AgentDeps`
3. Pydantic AI agent calls tools as needed (CLAP search, CNN similarity, key compatibility, sample analysis, song context, uploads, pair presentation, verdicts, kit building, preferences)
4. Agent streams SSE response (Vercel AI SDK format)
5. Frontend renders streamed chunks with tool-call transparency and interactive code fence blocks
6. Upload flow: `POST /samples/upload` → validate, analyze, embed → post-upload dialog for corrections → agent tools for finding/comparing uploads

## Additional Instructions

-   This project uses Pydantic AI. Documentation is available at `docs/pydantic-ai-llms-full.txt`. Read this file when working on agent code or when you need Pydantic AI API reference. Re-download periodically with `curl -o docs/pydantic-ai-llms-full.txt https://ai.pydantic.dev/llms-full.txt`.
-   Vercel AI SDK UI documentation is available at `docs/vercel-ai-sdk-ui.txt`. This project only uses **AI SDK UI** (hooks like `useChat` for chat UI) — it does NOT use AI SDK Core (LLM orchestration is handled by Pydantic AI on the backend). Read this file when working on frontend chat UI, message rendering, `useChat` hook, or streaming integration. Re-download with: `curl -s https://ai-sdk.dev/llms.txt | awk '/^# AI SDK UI$/{if(!found){found=1; printing=1}} /^# AI_APICallError$/{if(printing){printing=0; exit}} printing' > docs/vercel-ai-sdk-ui.txt`.
-   Assume that Git operations for branches, commits, and pushes will mostly be done manually. If executing a multi-step, comprehensive plan that involves successive commits, ask before making a commit.
-   All commands in this file are designed to run from the repo root. Do not use `cd <dir> && ...` patterns -- use `--directory` (uv) or `-C` (pnpm) flags instead.
-   Do not make any changes until you have 95% confidence that you know what to build -- ask me follow up questions using the AskUserQuestion tool until you have that confidence; but don't ask obvious questions, dig into the hard parts I might not have considered.
-   Do not worry about running the pytest commands yet. I have not implemented unit tests and likely will not for a while.
-   After modifying backend API endpoints, regenerate the frontend client with `pnpm -C frontend generate-client`. Do not manually edit files in `frontend/api/generated/`.
-   Audio sample files live in your local sample library (configured via `SAMPLE_LIBRARY_DIR` in `.env`)
-   Audio transforms (pitch-shift/time-stretch) use the Rubber Band CLI (`rubberband --fine`, R3 engine) via subprocess
-   CLAP model is loaded at startup via lifespan. CNN model is also loaded at startup if a checkpoint exists. Mock both in tests.

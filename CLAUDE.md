# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See @README.md for a project overview and @DEVELOPMENT.md for a development guide.

When working with this codebase, prioritize readability over cleverness. Ask clarifying questions before making architectural changes.

## Common Commands

### Backend (Python)

```bash
# Install dependencies
cd backend && uv sync

# Run backend with Docker (includes PostgreSQL)
docker compose up -d

# Pre-commit hooks (covers type checking, linting, and formatting for backend)
uv run pre-commit run --all-files

# Create local database migration
cd backend && ./scripts/create-db-revision-docker.sh "<migration_message>"

# Apply pending migrations (ask user first)
cd backend && ./scripts/migrate-docker.sh
```

### Frontend (TypeScript/Next.js)

```bash
cd frontend && pnpm install
cd frontend && pnpm lint            # lint with ultracite
cd frontend && pnpm format          # format with ultracite
cd frontend && pnpm generate-client # regenerate API client from backend OpenAPI
```

After making frontend code changes, run `pnpm format` to fix formatting. Use `pnpm lint` to check for errors. Do not run `pnpm build` for validation -- it's slow and rarely catches issues that linting misses. The dev server (`pnpm dev`) is typically already running during development.

## Architecture

### Backend (`backend/`)

FastAPI Python backend using async patterns throughout.

-   **`src/samplespace/app.py`**: FastAPI application entry point with CORS middleware and lifespan handler (CLAP model loading)
-   **`src/samplespace/routers/`**: API routes by domain (samples, agent, health)
-   **`src/samplespace/agents/`**: Pydantic AI agent -- `sample_agent.py` defines the sample assistant agent with tools for CLAP search, CNN similarity, key compatibility, sample analysis, and song context management; `deps.py` defines shared `AgentDeps` (includes `thread_id` and `song_context`); `tools/` contains agent tools (`clap_tools.py`, `cnn_tools.py`, `analysis_tools.py`, `context_tools.py`)
-   **`src/samplespace/models/`**: SQLAlchemy async models with CRUD classmethods (Sample with pgvector embedding columns)
-   **`src/samplespace/schemas/`**: Pydantic schemas for API contracts
-   **`src/samplespace/services/`**: Business logic (audio analysis, CLAP embedding generation, sample management)
-   **`src/samplespace/ml/`**: PyTorch CNN -- model definition (`model.py`), torchaudio dataset (`dataset.py`), training script (`train.py`), inference wrapper (`predict.py`)
-   **`src/samplespace/core/config.py`**: Settings via pydantic-settings (reads from `.env`)
-   **`src/samplespace/migrations/`**: Alembic migrations for PostgreSQL + pgvector
-   **`src/samplespace/dependencies/`**: FastAPI dependency injection (db sessions, OpenAI client)

See `.claude/rules/backend/code-conventions.md` for code style and conventions.

### Frontend (`frontend/`)

Next.js 16 with App Router.

-   **`app/page.tsx`**: Main sample browser page
-   **`app/api/chat/route.ts`**: Proxy route that forwards chat requests to backend agent
-   **`components/chat-panel.tsx`**: Chat component using `@ai-sdk/react` useChat hook; fetches and passes song context to header
-   **`components/song-context-badge.tsx`**: Read-only badge displaying active song context (key/BPM/genre/vibe) as pills
-   **`components/sample-browser.tsx`**: Sample grid with key/BPM/type filters
-   **`components/audio-player.tsx`**: Audio playback controls
-   **`components/waveform-viz.tsx`**: wavesurfer.js waveform rendering
-   **`api/client.ts`**: Axios client configuration (baseURL, credentials)
-   **`api/hooks/`**: Custom TanStack Query hooks wrapping generated client
-   **`api/generated/`**: Auto-generated TypeScript client from OpenAPI (do not edit manually)
-   **`components/ui/`**: Reusable UI components (shadcn/ui style)

Key patterns:

-   No auth currently -- planned for future
-   Uses Vercel AI SDK's `useChat` for streaming chat
-   Backend URL configured via `NEXT_PUBLIC_BACKEND_URL` env var
-   TanStack Query for data fetching with automatic caching/invalidation
-   Generated API client from backend OpenAPI schema -- run `pnpm generate-client` after backend API changes

### Data Flow

1. Frontend `useChat` sends messages to `/api/chat` route; non-streaming calls use TanStack Query hooks from `api/hooks/`
2. Route proxies raw request body to backend `POST /agent/chat`
3. Backend loads thread's `song_context` (if any) and injects it into `AgentDeps`
4. Pydantic AI agent decides which tools to call:
    - `search_by_description()` -- CLAP text-to-audio semantic search via pgvector (enriched with song context vibe)
    - `find_similar_samples()` -- CNN embedding nearest neighbors via pgvector
    - `check_key_compatibility()` -- circle of fifths / music theory logic
    - `analyze_sample()` -- full metadata retrieval (key, BPM, duration, type)
    - `suggest_complement()` -- combines CLAP search + key/BPM filtering (uses song context as fallback)
    - `set_song_context()` -- persists key/BPM/genre/vibe to thread for context-aware searches
5. Agent streams response back as SSE (Vercel AI SDK format)
6. Frontend renders streamed chunks with tool-call transparency; song context badge updates in chat header

## Additional Instructions

-   This project uses Pydantic AI. Documentation is available at `docs/pydantic-ai-llms-full.txt`. Read this file when working on agent code or when you need Pydantic AI API reference. Re-download periodically with `curl -o docs/pydantic-ai-llms-full.txt https://ai.pydantic.dev/llms-full.txt`.
-   Vercel AI SDK UI documentation is available at `docs/vercel-ai-sdk-ui.txt`. This project only uses **AI SDK UI** (hooks like `useChat` for chat UI) — it does NOT use AI SDK Core (LLM orchestration is handled by Pydantic AI on the backend). Read this file when working on frontend chat UI, message rendering, `useChat` hook, or streaming integration. Re-download with: `curl -s https://ai-sdk.dev/llms.txt | awk 'BEGIN{n=0} /^# AI SDK UI$/{n++} n==2 && /^# AI SDK UI$/,/^# AI_APICallError$/{if(/^# AI_APICallError$/) next; print}' > docs/vercel-ai-sdk-ui.txt`.
-   Assume that Git operations for branches, commits, and pushes will mostly be done manually. If executing a multi-step, comprehensive plan that involves successive commits, ask before making a commit.
-   Do not run `cd ... && git ...` for git commands in this repo. Assume you are already in the codebase.
-   Do not make any changes until you have 95% confidence that you know what to build -- ask me follow up questions using the AskUserQuestion tool until you have that confidence; but don't ask obvious questions, dig into the hard parts I might not have considered.
-   Do not worry about running the pytest commands yet. I have not implemented unit tests and likely will not for a while.
-   After modifying backend API endpoints, regenerate the frontend client with `cd frontend && pnpm generate-client`. Do not manually edit files in `frontend/api/generated/`.
-   Audio sample files in `data/samples/` are gitignored -- use `cd backend && uv run seed-db` to populate.
-   CLAP model is ~600MB, loaded at startup via lifespan. Mock in tests.
-   CNN training data is small (50-100 samples) -- the architecture and pipeline matter more than results.

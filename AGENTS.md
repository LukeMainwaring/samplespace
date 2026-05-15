# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex, etc.) when working with code in this repository.

For a project overview, features, and demo workflows, read `README.md`. For setup, commands, training workflows, and database migrations, read `DEVELOPMENT.md`.

When working with this codebase, prioritize readability over cleverness. Ask clarifying questions before making architectural changes.

## Common Commands

All commands in this file are designed to run from the repo root. Do not use `cd <dir> && ...` patterns -- use `--directory` (uv) or `-C` (pnpm) flags instead.

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

### Frontend (`frontend/`)

Next.js 16 with App Router; the chat UI lives under the `(chat)` route group.

-   **`app/(chat)/page.tsx`**: Main chat page (sample browser is at `app/(chat)/samples/page.tsx`, candidate uploads at `app/(chat)/candidates/page.tsx`)
-   **`app/(chat)/api/chat/route.ts`**: Thin proxy that forwards chat requests to backend `POST /api/agent/chat`
-   **`components/chat.tsx`**: Chat orchestrator using `@ai-sdk/react` `useChat` (typed as `ChatMessage` from `lib/types.ts`); fetches song context, manages file attachments, wraps messages in `ChatActionsProvider` so verdict/kit buttons can call `sendMessage` without prop drilling
-   **`components/message.tsx`**: Per-message renderer; `DataPartRenderer` switches on `part.type === "data-<name>"` to render interactive blocks (sample-results, kit, kit-preview, audio, pair-verdict)
-   **`components/sample-browser.tsx`** + **`components/sample-detail-panel.tsx`**: Split-pane sample grid + Splice-style detail panel (full metadata, waveform, mel spectrogram, CNN-similar samples) driven by `selectedSampleId`
-   **`components/elements/`**: Reusable building blocks — `sample-card.tsx` is the shared card; `sample-results-block.tsx` / `kit-block.tsx` / `kit-preview-block.tsx` / `audio-block.tsx` / `pair-verdict-block.tsx` are the data-part renderers; `tool-call.tsx` is the generic tool-call transparency UI
-   **`api/hooks/`**: Custom TanStack Query hooks wrapping the generated client (`api/generated/` — do not edit by hand)
-   **`components/ui/`**: shadcn/ui primitives

### Data Flow

1. Frontend `useChat` → `/api/chat` route → proxied to backend `POST /api/agent/chat`
2. Backend loads thread's `song_context` and injects into `AgentDeps`
3. Pydantic AI agent calls tools as needed (CLAP search, CNN similarity, key compatibility, sample analysis, song context, uploads, pair presentation, verdicts, kit building, preferences)
4. Agent streams SSE response (Vercel AI SDK format)
5. Frontend renders streamed chunks with tool-call transparency and interactive code fence blocks
6. Upload flow: `POST /samples/upload` → validate, analyze, embed → post-upload dialog for corrections → agent tools for finding/comparing uploads

## Additional Instructions

-   Editing backend Python? See `.claude/rules/backend/code-conventions.md` first — filenames, typing, FastAPI/SQLAlchemy/Pydantic patterns, `__init__.py` re-export convention.
-   Editing the Pydantic AI agent or evals? See `.claude/rules/backend/pydantic-ai.md` first — docs are split between the local snapshot and `ai.pydantic.dev`; tool error-handling convention (let exceptions propagate to `hooks._recover_tool_error`).
-   Editing frontend code? See `.claude/rules/frontend/code-conventions.md` first — `@/` import alias scope, shadcn/ui usage, kebab-case filenames, `cn()` from `@/lib/utils`.
-   Editing the chat UI or anything `useChat`-adjacent? See `.claude/rules/frontend/vercel-ai-sdk.md` first — pinned UI docs cover the UI surface only; everything else on `ai-sdk.dev`. No AI SDK Core server-side. `useChat` is on the v5/v6 `DefaultChatTransport` + `sendMessage` shape. Renderable interactive blocks come in on `data-<name>` parts, dispatched in `message.tsx::DataPartRenderer`.
-   Assume that Git operations for branches, commits, and pushes will mostly be done manually. If executing a multi-step, comprehensive plan that involves successive commits, ask before making a commit.
-   After modifying backend API endpoints, regenerate the frontend client with `pnpm -C frontend generate-client`. Do not manually edit files in `frontend/api/generated/`.
-   Audio sample files live in your local sample library (configured via `SAMPLE_LIBRARY_DIR` in `.env`).
-   Audio transforms (pitch-shift/time-stretch) use the Rubber Band CLI (`rubberband --fine`, R3 engine) via subprocess. Requires `rubberband` on PATH (see DEVELOPMENT.md Setup for install).
-   CLAP model is loaded at startup via lifespan. CNN model is also loaded at startup if a checkpoint exists. Mock both in tests.

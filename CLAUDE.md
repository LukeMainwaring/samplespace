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

FastAPI Python backend using async patterns throughout.

-   **`src/samplespace/app.py`**: FastAPI application entry point with CORS middleware and lifespan handler (CLAP + CNN model loading)
-   **`src/samplespace/routers/`**: API routes by domain (samples, agent, health)
-   **`src/samplespace/agents/`**: Pydantic AI agent -- `sample_agent.py` defines the sample assistant agent with tools for CLAP search, CNN similarity, key compatibility, sample analysis, song context management, upload similarity, pair presentation, verdict recording, kit building, and preference learning; `deps.py` defines shared `AgentDeps` (includes `thread_id` and `song_context`); `tools/` contains agent tools (`clap_tools.py`, `cnn_tools.py`, `analysis_tools.py`, `context_tools.py`, `pair_tools.py`, `transform_tools.py`, `upload_tools.py`, `verdict_tools.py`, `kit_tools.py`, `preference_tools.py`, `formatting.py`)
-   **`src/samplespace/models/`**: SQLAlchemy async models with CRUD classmethods (Sample with pgvector embedding columns, PairVerdict, Thread)
-   **`src/samplespace/schemas/`**: Pydantic schemas for API contracts
-   **`src/samplespace/services/`**: Business logic (audio analysis, CLAP embedding generation, sample management, upload processing, pair scoring, pair feature extraction, music theory, kit building, spectrogram generation, preference model training/prediction, shared candidate search utilities)
-   **`src/samplespace/ml/`**: PyTorch CNN (4 residual blocks, SE attention, 1→64→128→256→512 channels, 2-layer projection head) -- model definition (`model.py`), torchaudio dataset with waveform augmentation (polarity inversion, speed/pitch perturbation via fast resample, noise, EQ) and spectrogram augmentation (time/freq masking, gain) (`dataset.py`), training script with SupCon + cross-entropy loss, mixup, class-weighted sampling, cosine annealing, mixed precision, TensorBoard logging (`train.py`), inference wrapper with batch support (`predict.py`)
-   **`src/samplespace/core/config.py`**: Settings via pydantic-settings (reads from `.env`)
-   **`src/samplespace/migrations/`**: Alembic migrations for PostgreSQL + pgvector
-   **`src/samplespace/dependencies/`**: FastAPI dependency injection (db sessions, OpenAI client, CLAP models, CNN model)

See `.claude/rules/backend/code-conventions.md` for code style and conventions.

### Frontend (`frontend/`)

Next.js 16 with App Router.

-   **`app/page.tsx`**: Main sample browser page
-   **`app/api/chat/route.ts`**: Proxy route that forwards chat requests to backend agent
-   **`components/chat.tsx`**: Chat orchestrator using `@ai-sdk/react` useChat hook; fetches and passes song context to header; manages file attachment state for uploads; wraps messages with `ChatActionsProvider` for verdict buttons
-   **`components/messages.tsx`**: Message list container with smart scroll behavior (MutationObserver/ResizeObserver-based auto-scroll, scroll-to-bottom button)
-   **`components/message.tsx`**: Individual message rendering (`PreviewMessage`) and loading state (`RiffingMessage`)
-   **`components/multimodal-input.tsx`**: Chat input with file attachment (paperclip button), local storage persistence, auto-focus, and memoization
-   **`components/greeting.tsx`**: Animated empty state with Framer Motion fade-in
-   **`components/song-context-badge.tsx`**: Read-only badge displaying active song context (key/BPM/genre/vibe) as pills
-   **`components/sample-browser.tsx`**: Sample grid with key/BPM/type filters; split-pane layout driven by `selectedSampleId` — when a sample is selected, the list compresses to the left and a detail panel appears on the right
-   **`components/sample-detail-panel.tsx`**: Splice-style inline detail panel showing full metadata, waveform, mel spectrogram (full/CNN toggle), and CNN-similar samples with similarity percentages; manages its own playback state independently from the sample list
-   **`components/candidate-samples.tsx`**: Upload page for reference tracks with CLAP similarity search
-   **`components/preview-attachment.tsx`**: File attachment chip with loading/complete states for chat input
-   **`components/elements/sample-card.tsx`**: Shared sample card component (filename, metadata pills, WaveformViz) used by pair-verdict-block, kit-block, and sample-results-block
-   **`components/elements/sample-results-block.tsx`**: Renders `sample-results` code fences as a vertical list of playable SampleCards (used by all search/similarity tools)
-   **`components/audio-player.tsx`**: Audio playback controls
-   **`components/waveform-viz.tsx`**: wavesurfer.js waveform rendering
-   **`api/client.ts`**: Generated client configuration (baseURL from `lib/constants.ts`, credentials)
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
    - `find_similar_to_upload()` -- CLAP audio-to-audio search using an uploaded sample's embedding (excludes other uploads)
    - `present_pair()` -- finds a complementary candidate via CLAP search (with song context) or CNN similarity, supports random anchors for rapid pairing, and returns a `pair-verdict` code fence with mixed preview
    - `record_verdict()` -- persists user's thumbs up/down verdict, fires background relational feature extraction, and triggers preference model retraining when threshold is met
    - `show_preferences()` -- surfaces learned feature importances as natural-language explanations
    - `build_kit()` -- assembles a multi-sample kit via greedy pairwise optimization (CLAP retrieval per type, fast inline scoring, CNN diversity penalty) and returns a `kit` code fence
5. Agent streams response back as SSE (Vercel AI SDK format)
6. Frontend renders streamed chunks with tool-call transparency; song context badge updates in chat header; `sample-results` code fences render as playable sample cards; `pair-verdict` code fences render as interactive side-by-side audio players with mixed preview, verdict buttons, and "Next Pair" for rapid sessions; `kit` code fences render as kit cards with per-slot playback and compatibility scores
7. Upload flow: frontend `POST /samples/upload` → backend validates, stores in `backend/data/uploads/`, analyzes, generates CLAP embedding → returns sample metadata + ID → user references ID in chat → agent calls `find_similar_to_upload`

## Additional Instructions

-   This project uses Pydantic AI. Documentation is available at `docs/pydantic-ai-llms-full.txt`. Read this file when working on agent code or when you need Pydantic AI API reference. Re-download periodically with `curl -o docs/pydantic-ai-llms-full.txt https://ai.pydantic.dev/llms-full.txt`.
-   Vercel AI SDK UI documentation is available at `docs/vercel-ai-sdk-ui.txt`. This project only uses **AI SDK UI** (hooks like `useChat` for chat UI) — it does NOT use AI SDK Core (LLM orchestration is handled by Pydantic AI on the backend). Read this file when working on frontend chat UI, message rendering, `useChat` hook, or streaming integration. Re-download with: `curl -s https://ai-sdk.dev/llms.txt | awk '/^# AI SDK UI$/{if(!found){found=1; printing=1}} /^# AI_APICallError$/{if(printing){printing=0; exit}} printing' > docs/vercel-ai-sdk-ui.txt`.
-   Assume that Git operations for branches, commits, and pushes will mostly be done manually. If executing a multi-step, comprehensive plan that involves successive commits, ask before making a commit.
-   All commands in this file are designed to run from the repo root. Do not use `cd <dir> && ...` patterns -- use `--directory` (uv) or `-C` (pnpm) flags instead.
-   Do not make any changes until you have 95% confidence that you know what to build -- ask me follow up questions using the AskUserQuestion tool until you have that confidence; but don't ask obvious questions, dig into the hard parts I might not have considered.
-   Do not worry about running the pytest commands yet. I have not implemented unit tests and likely will not for a while.
-   After modifying backend API endpoints, regenerate the frontend client with `pnpm -C frontend generate-client`. Do not manually edit files in `frontend/api/generated/`.
-   Audio sample files live in your local sample library (configured via `SAMPLE_LIBRARY_DIR` in `.env`) -- use `uv run --directory backend seed-samples` to populate the database.
-   CLAP model is ~600MB, loaded at startup via lifespan. CNN model is also loaded at startup if a checkpoint exists. Mock both in tests.
-   CNN training defaults: 100 epochs, batch size 64, mixup (alpha 0.2), class-weighted sampling, cosine annealing with 5-epoch warmup, early stopping (patience 15). Run `uv run --directory backend train-cnn --help` for all options. TensorBoard logs go to `backend/data/runs/`.

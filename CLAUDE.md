# CLAUDE.md

## Overview

SampleSpace — a multi-modal AI-powered tool for music producers to discover and match audio samples. Combines a custom PyTorch CNN for spectrogram-based similarity, CLAP embeddings for natural language audio search, and a Pydantic AI agent that orchestrates both.

This is a portfolio/resume showcase project, not a production product.

## Tech Stack

- **Frontend:** Next.js 15 (App Router) + Tailwind + shadcn/ui
- **Backend:** FastAPI + Pydantic v2 + async SQLAlchemy
- **ML:** PyTorch + torchaudio (CNN), CLAP (`laion/larger_clap_music`) for embeddings
- **Agent:** Pydantic AI with tool-use architecture
- **Database:** PostgreSQL + pgvector
- **Audio:** librosa (analysis), torchaudio (ML transforms), music21 (key detection)
- **DevOps:** Docker Compose, GitHub Actions CI

## Project Structure

Monorepo with two main services:
- `frontend/` — Next.js app
- `backend/` — FastAPI app (includes ML model, CLAP embeddings, and agent in-process)
- `scripts/` — Seed data, batch embedding
- `data/` — Audio samples (gitignored)

## Development

```bash
docker compose up          # Start full stack (Postgres, backend, frontend)
python scripts/seed.py     # Populate sample data
python scripts/embed_samples.py  # Generate CLAP embeddings
```

## Key Patterns

- Backend uses `src/samplespace/` package layout with FastAPI app factory
- Pydantic AI agent has tools for: CLAP search, CNN similarity, key compatibility, sample analysis
- CLAP model loaded once at startup, kept in memory
- CNN is dual-head: classification logit + 128-dim embedding for similarity
- pgvector stores both CLAP (512-dim) and CNN (128-dim) embeddings
- Frontend uses Vercel AI SDK `useChat` for streaming agent responses

## Testing

```bash
cd backend && pytest       # Backend tests
cd frontend && npm test    # Frontend tests (Vitest)
```

## Important Notes

- Audio sample files in `data/samples/` are gitignored — use `seed.py` to populate
- CLAP model is ~600MB, loaded at startup. Mock in tests.
- CNN training data is small (50-100 samples) — the architecture/pipeline matters more than results
- See `PLAN.md` for the full implementation plan and design rationale

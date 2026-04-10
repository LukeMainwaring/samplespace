# Development

## Tools

This project uses:

-   **[uv]** - Python package installer and resolver
-   **[Docker]** - Containers for PostgreSQL + backend
-   **[Ruff]** - Python linter and formatter
-   **[mypy]** - Static type checker
-   **[pre-commit]** - Git hook framework (runs Ruff + mypy automatically on commit)
-   **[Rubber Band]** - Audio pitch-shifting and time-stretching (R3 engine via CLI)

[uv]: https://docs.astral.sh/uv/
[Docker]: https://docs.docker.com/get-docker/
[Ruff]: https://github.com/astral-sh/ruff
[mypy]: https://mypy-lang.org/
[pre-commit]: https://pre-commit.com/
[Rubber Band]: https://breakfastquay.com/rubberband/

## Setup

Install system dependencies:

```bash
# macOS
brew install rubberband

# Linux (Debian/Ubuntu)
apt install rubberband-cli
```

Install Python dependencies and pre-commit hooks:

```bash
uv sync --directory backend
uv run --directory backend pre-commit install
```

## Code Quality

Pre-commit hooks run Ruff formatting, Ruff linting, and mypy type checking automatically on every commit. To run manually:

```bash
uv run --directory backend pre-commit run --all-files
```

Individual tools:

```bash
uv run --directory backend ruff format .         # format
uv run --directory backend ruff check --fix .    # lint + autofix
uv run --directory backend mypy --strict src tests  # type check
```

## Database Migrations

```bash
# Create a new migration
./backend/scripts/create-db-revision-docker.sh "<migration_message>"

# Apply pending migrations
./backend/scripts/migrate-docker.sh

# Roll back one migration (use with caution)
./backend/scripts/downgrade-db-revision-docker.sh
```

## API Client Generation

The frontend uses a generated TypeScript client from the backend's OpenAPI schema.

After modifying backend API endpoints:

```bash
# Ensure backend is running
docker compose up -d

# Regenerate (fetches schema, generates types + hooks, formats)
pnpm -C frontend generate-client
```

Do not manually edit files in `frontend/api/generated/`. Custom hooks in `api/hooks/` wrap the generated code.

## Testing

```bash
uv run --directory backend pytest                 # unit tests (eval suite excluded)
uv run --directory backend pytest -m eval         # real-model sample_agent eval suite (opt-in)
```

The `eval` marker gates tests that call the real OpenAI API via `sample_agent` — the default `pytest` invocation excludes them via `addopts = "-m 'not eval'"`. Use them as a nightly safety net on `main` or manual spot-checks, not on every PR. See `.claude/rules/backend/pydantic-ai.md` and `backend/tests/evals/` for the suite layout.

## Audio Data

Audio sample files are gitignored. Set `SAMPLE_LIBRARY_DIR` in `.env` to your local sample library, then:

```bash
uv run --directory backend seed-samples       # populate database from library
uv run --directory backend embed-samples      # generate CLAP embeddings (~2 min)
uv run --directory backend embed-cnn          # generate CNN embeddings (after training)
```

## ML Development

### CNN Training

```bash
uv run --directory backend train-cnn
uv run --directory backend train-cnn --help  # all options
```

Checkpoints saved to `backend/data/checkpoints/`. TensorBoard logs to `backend/data/runs/`:

```bash
uv run --directory backend tensorboard --logdir backend/data/runs/
```

### Preference Model

Logistic regression on 10-dimensional pair verdict features (4 pair scores + 6 relational audio features).

```bash
uv run --directory backend train-preferences
```

Artifacts saved to `backend/data/models/`. Auto-retrains in the background every 5th verdict after 15 verdicts.

## Continuous Integration

Testing, type checking, and formatting/linting is [checked in CI][ci].

[ci]: .github/workflows/ci.yml

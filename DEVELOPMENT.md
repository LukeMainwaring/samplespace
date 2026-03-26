# Development

## Tools

This project uses:

-   **[uv]** - Fast Python package installer and resolver for dependency management
-   **[Docker]** - Container platform for local development and production deployment
-   **[Ruff]** - Fast Python linter and formatter
-   **[mypy]** - Static type checker
-   **[pre-commit]** - Git hook framework for automated code quality checks

[uv]: https://docs.astral.sh/uv/
[Docker]: https://docs.docker.com/get-docker/
[Ruff]: https://github.com/astral-sh/ruff
[mypy]: https://mypy-lang.org/
[pre-commit]: https://pre-commit.com/

## Setup

Install dependencies:

```bash
uv sync
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

## Pre-commit hooks

We use [pre-commit] to automatically run linting, formatting, and type checking on all commits.

To manually check all files:

```bash
uv run pre-commit run --all-files
```

The hooks will run automatically when you commit. If any checks fail, the commit will be blocked and files will be auto-fixed where possible. Review the changes and commit again.

**Note:** Pre-commit hooks require backend dependencies to be installed first (`cd backend && uv sync`).

## Testing

Run the tests:

```bash
uv run pytest
```

Run specific test markers:

```bash
uv run pytest -m main
uv run pytest -m additional
```

[pytest-mark]: https://docs.pytest.org/en/stable/example/markers.html

## Type checking

Type checking with [mypy] runs automatically via pre-commit hooks.

To manually run the type checker:

```bash
uv run mypy --strict src tests
```

## Formatting and linting

Formatting and linting with [Ruff] runs automatically via pre-commit hooks.

To manually run the formatter and linter:

```bash
# Format and fix issues
uv run ruff format .
uv run ruff check --fix .

# Check only (no modifications)
uv run ruff format --check .
uv run ruff check .
```

## Continuous integration

Testing, type checking, and formatting/linting is [checked in CI][ci].

[ci]: .github/workflows/ci.yml

## Database migrations

```bash
# Create a new migration (generates file in migrations/versions/)
cd backend && ./scripts/create-db-revision-docker.sh "<migration_message>"

# Apply all pending migrations
cd backend && ./scripts/migrate-docker.sh

# Roll back one migration (use with caution -- may cause data loss)
cd backend && ./scripts/downgrade-db-revision-docker.sh
```

## API Client Generation

The frontend uses a generated TypeScript client from the backend's OpenAPI schema.

After modifying backend API endpoints:

```bash
# Ensure backend is running
docker compose up -d

# Regenerate client (fetches schema, generates types, formats)
cd frontend && pnpm generate-client
```

This generates:

-   `api/generated/types.gen.ts` - TypeScript types from OpenAPI schemas
-   `api/generated/sdk.gen.ts` - API functions for each endpoint
-   `api/generated/@tanstack/react-query.gen.ts` - TanStack Query hooks

**Do not manually edit files in `frontend/api/generated/`** -- they are overwritten on regeneration.

Custom hooks in `api/hooks/` wrap the generated code with cleaner APIs.

## Audio Data

Audio sample files are gitignored. To populate:

```bash
# Seed sample data (place audio files in data/samples/, organized by subdirectory)
cd backend && uv run seed-db

# Generate CLAP embeddings for all seeded samples
cd backend && uv run embed-samples
```

Audio files are stored in `data/samples/`. The database stores metadata (key, BPM, duration, type) and embedding vectors (CLAP 512-dim, CNN 128-dim).

## ML Development

### CNN Training

```bash
python -m samplespace.ml.train
```

Model checkpoints are saved to `data/checkpoints/` (gitignored). The training set is small (50-100 samples) -- the architecture and pipeline are the focus, not benchmark results.

### CLAP Model

The CLAP model (`laion/clap-htsat-unfused`) is ~600MB and cached by HuggingFace transformers in `~/.cache/huggingface/`. It is loaded once at startup via the FastAPI lifespan handler. Mock it in tests.

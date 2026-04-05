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
uv sync --directory backend
```

Install pre-commit hooks:

```bash
uv run --directory backend pre-commit install
```

## Pre-commit hooks

We use [pre-commit] to automatically run linting, formatting, and type checking on all commits.

To manually check all files:

```bash
uv run --directory backend pre-commit run --all-files
```

The hooks will run automatically when you commit. If any checks fail, the commit will be blocked and files will be auto-fixed where possible. Review the changes and commit again.

**Note:** Pre-commit hooks require backend dependencies to be installed first (`uv sync --directory backend`).

## Testing

Run the tests:

```bash
uv run --directory backend pytest
```

Run specific test markers:

```bash
uv run --directory backend pytest -m main
uv run --directory backend pytest -m additional
```

[pytest-mark]: https://docs.pytest.org/en/stable/example/markers.html

## Type checking

Type checking with [mypy] runs automatically via pre-commit hooks.

To manually run the type checker:

```bash
uv run --directory backend mypy --strict src tests
```

## Formatting and linting

Formatting and linting with [Ruff] runs automatically via pre-commit hooks.

To manually run the formatter and linter:

```bash
# Format and fix issues
uv run --directory backend ruff format .
uv run --directory backend ruff check --fix .

# Check only (no modifications)
uv run --directory backend ruff format --check .
uv run --directory backend ruff check .
```

## Continuous integration

Testing, type checking, and formatting/linting is [checked in CI][ci].

[ci]: .github/workflows/ci.yml

## Database migrations

```bash
# Create a new migration (generates file in migrations/versions/)
./backend/scripts/create-db-revision-docker.sh "<migration_message>"

# Apply all pending migrations
./backend/scripts/migrate-docker.sh

# Roll back one migration (use with caution -- may cause data loss)
./backend/scripts/downgrade-db-revision-docker.sh
```

## API Client Generation

The frontend uses a generated TypeScript client from the backend's OpenAPI schema.

After modifying backend API endpoints:

```bash
# Ensure backend is running
docker compose up -d

# Regenerate client (fetches schema, generates types, formats)
pnpm -C frontend generate-client
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
# Seed sample data (set SAMPLE_LIBRARY_DIR in .env to your local sample library)
uv run --directory backend seed-samples

# Generate CLAP embeddings for all seeded samples
uv run --directory backend embed-samples
```

Audio files live in your local sample library directory (configured via `SAMPLE_LIBRARY_DIR`). The database stores metadata (key, BPM, duration, type) and embedding vectors (CLAP 512-dim, CNN 128-dim).

## ML Development

### CNN Training

```bash
uv run --directory backend train-cnn
uv run --directory backend train-cnn --epochs 50 --batch-size 32 --grad-accum 2
uv run --directory backend train-cnn --help  # all options
```

Model checkpoints are saved to `backend/data/checkpoints/` (gitignored). TensorBoard logs go to `backend/data/runs/` (gitignored). Defaults: 100 epochs, batch size 64, cosine annealing with 5-epoch linear warmup, early stopping (patience 15), mixed precision on CUDA.

Monitor training:

```bash
uv run --directory backend tensorboard --logdir backend/data/runs/
```

### Preference Model

The preference model learns pairing taste from pair verdicts. It trains automatically in the background after every 5th verdict (starting at 15 verdicts).

```bash
# Manual training
uv run --directory backend train-preferences
```

Model artifacts are saved to `backend/data/models/` (gitignored): `preference_model.joblib` (sklearn pipeline) and `preference_meta.json` (version, accuracy, feature importances). The model is a `Pipeline(StandardScaler, LogisticRegression)` trained on 10-dimensional feature vectors (4 pair scores + 6 relational audio features).

### CLAP Model

The CLAP model (`laion/clap-htsat-unfused`) is ~600MB and cached by HuggingFace transformers in `~/.cache/huggingface/`. It is loaded once at startup via the FastAPI lifespan handler. Mock it in tests.

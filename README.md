# Repository Template

- Pre-requisites: uv, python3.11

## Setup

```bash
# Install dependencies
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```

## Development

```bash
# Run linting and formatting
uv run ruff check --fix .

# Run formatting only
uv run ruff format .

# Run linting only (without fixes)
uv run ruff check .

# Run docstring coverage check
uv run interrogate .
```

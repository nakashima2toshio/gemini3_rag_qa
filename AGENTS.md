# Repository Guidelines

This repository implements a Japanese RAG (Retrieval‑Augmented Generation) Q&A system using OpenAI embeddings and Qdrant. Follow these guidelines to keep contributions consistent and easy to review.

## Project Structure & Module Organization
- Core modules (root): `rag_qa.py` (SemanticCoverage), `helper_api.py`, `helper_rag.py`, `helper_st.py`.
- Scripts: `make_qa.py` (analysis/demo), `server.py` (bootstrap + Qdrant checks).
- Docs and assets: `doc/` (design notes), `datasets/` (inputs), `OUTPUT/` (generated results).
- Infra: `docker-compose/` (Qdrant), `config.yml` (settings), `.env` (secrets; not committed).
- Legacy: `old_code/` (archived `aXX_*.py` utilities, including Streamlit UI).

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt` (or `python setup.py --quick`).
- Start Qdrant: `docker-compose -f docker-compose/docker-compose.yml up -d`.
- Run bootstrap: `python server.py` (starts/validates Qdrant; optional UI hooks).
- Streamlit UI (legacy): `streamlit run old_code/a50_rag_search_local_qdrant.py`.
- Lint/format: `ruff check .` and `ruff format`.

## Coding Style & Naming Conventions
- Python 3.8+; follow PEP 8, 4‑space indentation; prefer type hints.
- Names: functions/vars `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`, files `snake_case.py` (e.g., `helper_xx.py`).
- Keep modules focused; favor pure functions; include concise docstrings (triple double quotes).
- Avoid hard‑coded paths/keys; read from `.env` and `config.yml`.

## Testing Guidelines
- Framework: `pytest`. Place tests under `tests/` named `test_*.py`.
- Install: `pip install -U pytest pytest-cov` (or `uv add -d pytest pytest-cov`).
- Run tests: `pytest -q`. Target core logic first (e.g., `rag_qa.SemanticCoverage` and helpers).
- Use small local fixtures in `datasets/` or temporary data; avoid network calls.
- Coverage: `pytest -q --cov=rag_qa,helper_api,helper_rag --cov-report=term-missing --cov-fail-under=80` (target ≥ 80%).

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- PRs must include: clear description, linked issues, steps to verify (commands), and screenshots/logs for UI changes.
- Ensure `ruff check .` passes and update `README.md`/`doc/` when behavior or commands change.

## Security & Configuration Tips
- Do not commit secrets. Required envs: `OPENAI_API_KEY`, `QDRANT_URL` in `.env`.
- Prefer parametrized configs; keep sample data minimal and anonymized.

## CI Recommendations
- Use GitHub Actions to run `ruff check .` and `pytest` on PRs and pushes.
- Install deps and dev tools: `pip install -r requirements.txt && pip install -U pytest pytest-cov ruff`.
- Enforce coverage: `pytest --cov=rag_qa,helper_api,helper_rag --cov-report=term-missing --cov-fail-under=80`.
- Cache pip and test against Python 3.10/3.11/3.12 matrices.
- Upload `coverage.xml` as an artifact for visibility.
- For integration tests, start a Qdrant service container and wait on `/readyz`.

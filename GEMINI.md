# PROJECT CONSTITUTION

You MUST adhere to the following rules:

## 1. Environment & Execution
*   **Source of Truth**: The project uses **Pixi** (`pixi`) for dependency management.
*   **Execution**:
    *   ALWAYS use `pixi run <command>`.
    *   ALWAYS run modules via `pixi run python -m wolfgang_lm.<module>` to ensure correct package resolution.
    *   NEVER run modules via `python wolfgang_lm.<module>` or `python -m wolfgang_lm.<module>`.
*   **Dependencies**:
    *   NEVER install global pip packages.
    *   If a package is missing, add it to `pyproject.toml`.

## 2. Code Style & Quality
*   **Standards**: Adhere to PEP 8.
*   **Typing**: Strong preference for type hints (`def foo(x: int) -> str:`) in core model code.
*   **Comments (CRITICAL)**:
    *   **"HOW and WHY"**: For complex logic, you MUST write concise comments explaining *how* it works and *why* it is implemented that way.
    *   **Preservation**: **NEVER DELETE EXISTING COMMENTS.** If code is refactored, update the comments, but do not remove explanations unless they are undeniably obsolete and misleading.

## 3. Data Safety
*   **Restricted Areas**: `data*/` and `out-*/` directories contain massive binary/text files.
    *   **NEVER** read the full contents of these directories without filtering (use `ls` or `find` with limits).
    *   **NEVER** cat huge files from these directories.

## 4. Documentation
*   **Atomic Updates**: Code and Documentation are linked.
    *   If you change a CLI argument, function signature, or logic, you **MUST** update the correctly corresponding docs (e.g., `README.md`, `docs/*.md`) in the same turn.
    *   Search the docs/ folder to find the most relevant documentation for edited files.
    *   Keep the `README.md` technical specifications up to date.

## 5. Frontend & Serving
*   **Backend**: The API runs on **Uvicorn** (`uvicorn`).
    *   Command: `uvicorn wolfgang_lm.api.server:app --reload`
*   **Frontend**: The `web/` directory is Vanilla JS/HTML.
    *   **Simplicity**: Do not introduce complex build steps (No Webpack/Vite/React) unless explicitly requested.

## 6. Security & Configuration
*   **Secrets**:
    *   **NEVER** commit secrets (API keys, tokens, credentials).
    *   Use `.env` files and environment variables to handle sensitive data.
*   **Centralized Config**:
    *   **No Magic Values**: Avoid hardcoding parameters, paths, or magic numbers deep in the implementation code.
    *   **Centralization**: Use centralized configuration mechanisms (e.g., config classes/files) to define adjustable parameters.

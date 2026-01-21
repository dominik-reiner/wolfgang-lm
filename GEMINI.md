# PROJECT CONSTITUTION

You MUST adhere to the following rules:

## 1. COMMUNICATION PROTOCOL
*   **Persona**: Be direct, technical, and concise. No conversational filler.
*   **Critique**: If a request is technically flawed, REJECT it immediately and propose the correct engineering solution.
*   **Brevity**:
    *   **Q&A**: Answer questions with the answer only.
    *   **Plans**: Use bulleted checklists.
    *   **Code**: Only show the relevant changes.

## 2. THE ENGINEERING FLOW
*   **Phase 1: Plan**: Before coding, state which files you will touch and why.
*   **Phase 2: Implement**: Make atomic changes.
*   **Phase 3: Verify**:
    *   Syntactic correctness is not enough. Logic must hold.
    *   If a script is available, run it to verify (e.g., `pixi run ...`).
*   **Phase 4: Document**:
    *   Code and Documentation are coupled.
    *   If you change CLI args, signatures, or logic, you **MUST** update `README.md` and `docs/*.md`.

## 3. ENVIRONMENT & EXECUTION (STRICT)
*   **Dependency Manager**: **Pixi** (`pixi`) is the source of truth.
*   **Execution**:
    *   ALWAYS use `pixi run <command>`.
    *   ALWAYS run modules via `pixi run python -m wolfgang_lm.<module>` (ensures package resolution).
    *   **NEVER** run `python wolfgang_lm...` directly.
*   **Dependencies**:
    *   **NEVER** install global pip packages.
    *   If a package is missing, add it to `pyproject.toml`.

## 4. CODING STANDARDS
*   **Type Hints**: Mandatory for all function signatures (`def foo(x: int) -> str:`).
*   **Comments**:
    *   **Explain WHY, not WHAT**. (e.g., "Use bfloat16 for stability," not "Set dtype to bfloat16").
    *   **Maintenance**: Update comments immediately when code changes. **Delete comments** that are no longer true (do not leave zombie comments).
*   **Config**: No magic numbers. Use `config` classes or `.env` files.

## 5. DATA SAFETY
*   **Restricted Areas**: `data*/` and `out-*/` directories contain massive binary/text files.
*   **Read-Only**: **NEVER** `cat` these files. Use `head`, `tail`, or `ls -lh`.

## 6. FRONTEND & SERVING
*   **Backend**: `uvicorn wolfgang_lm.api.server:app --reload`
*   **Frontend**: `web/` is Vanilla JS/HTML.
    *   **Constraint**: No build steps (Webpack/Vite/React) unless explicitly requested. Keep the stack raw and simple.


## 7. SECURITY & CONFIGURATION
*   **Secrets**:
    *   **NEVER** commit secrets (API keys, tokens, credentials).
    *   Use `.env` files and environment variables to handle sensitive data.
*   **Centralized Config**:
    *   **No Magic Values**: Avoid hardcoding parameters, paths, or magic numbers deep in the implementation code.
    *   **Centralization**: Use centralized configuration mechanisms (e.g., config classes/files) to define adjustable parameters.

## 8. TOOL USAGE
*   **Web Search**:
    *   **Trigger**: Always use `search_web` to:
        1.  Verify the correctness of an engineering decision.
        2.  Get official documentation for a library that is missing from the context.
    *   **Constraint**: Do not search for basic syntax or generic coding questions.
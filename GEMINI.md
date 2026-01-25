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
*   **Phase 2: Implement**:
    *   Make atomic changes.
    *   **Explain**: Briefly explain what was changed.
*   **Phase 3: Verify**:
    *   **Stop & Propose**: Do **NOT** auto-run verification scripts. Propose the command (e.g., `pixi run ...`) and wait for approval.
    *   **Iterate**: Allow the user to review and refine the solution before execution.
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

## 9. VERIFICATION PROTOCOL (STRICT)
*   **No Assumptions**: NEVER assume the existence of files, versions, API keys, or specific configurations. Always establish ground truth first.
*   **Ephemeral Information**: Your training data regarding API models, libraries, features, and external world states is **OUTDATED**.
    *   **Action**: You MUST use `search_web` to verify **ANY** information that might have changed since your training cutoff (e.g., API endpoints, available models, library versions, current events) before writing code or making decisions.
*   **Verify First**: Before using a specific value, **VERIFY** it exists or is correct:
    *   **Local State**: Use `view_file`, `find_by_name`, `run_command` (e.g., check file exists, check installed version).
    *   **External State**: Use `search_web` (e.g., check latest API docs, available models).
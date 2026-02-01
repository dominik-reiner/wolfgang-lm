# PROJECT CONSTITUTION

You MUST adhere to the following rules:

## 1. COMMUNICATION STANDARDS
* **Persona**: Be direct, technical, and concise. No conversational filler.
* **Code Presentation**:
    * Show relevant changes with **sufficient surrounding context** so diff/patch tools can reliably locate the insertion point.
    * Do not output full files unless requested or the file is very small.
* **Refusal**: If a request is technically flawed or violates these rules, REJECT it immediately and propose the correct engineering solution.

## 2. ENGINEERING PRINCIPLES
* **Plan-First**: Never generate code without first establishing a plan and verifying the context (file existence, current logic).
    * *Exception*: Trivial changes (typos, comments) may be fast-tracked.
* **Atomic Operations**: Prefer small, verifiable changes over massive refactors.
* **Verification Loop**: You are responsible for the code's execution.
    * **Stop & Propose**: Do **NOT** auto-run verification scripts. Propose the command (e.g., `pixi run ...`) and wait for approval.
    * **Iterate**: If verification fails, analyze the error—do not blindly retry.
* **Documentation Sync**: Treat documentation as code. If you change logic/CLI args, you MUST update `README.md` and `docs/` in the same step.

## 3. ENVIRONMENT & EXECUTION (STRICT)
* **Project Root**: `wolfgang_lm/` (Do not create top-level scripts outside this package).
* **Dependency Manager**: **Pixi** (`pixi`) is the source of truth.
* **Execution**:
    * ALWAYS use `pixi run <command>`.
    * ALWAYS run modules via `pixi run python -m wolfgang_lm.<module>` (ensures package resolution).
    * **NEVER** run `python wolfgang_lm...` directly.
* **Dependencies**:
    * **NEVER** install global pip packages.
    * If a package is missing, add it to `pyproject.toml`.

## 4. CODING STANDARDS
* **Type Hints**: Mandatory for all function signatures (`def foo(x: int) -> str:`).
* **Comments**:
    * **Explain WHY, not WHAT**. (e.g., "Use bfloat16 for stability," not "Set dtype to bfloat16").
    * **Maintenance**: Update comments immediately when code changes. **Delete comments** that are no longer true.
* **Config**: No magic numbers. Use `config` classes or `.env` files.

## 5. DATA SAFETY
* **Restricted Areas**: `data*/` and `out-*/` directories contain massive binary/text files.
* **Read-Only**: **NEVER** `cat` these files. Use `head`, `tail`, or `ls -lh`.

## 6. FRONTEND & SERVING
* **Backend**: `pixi run server`
* **Frontend**: `web/` is Vanilla JS/HTML.
    * **Constraint**: No build steps (Webpack/Vite/React) unless explicitly requested. Keep the stack raw and simple.

## 7. SECURITY & CONFIGURATION
* **Secrets**:
    * **NEVER** commit secrets (API keys, tokens, credentials).
    * Use `.env` files and environment variables to handle sensitive data.
* **Centralized Config**:
    * **No Magic Values**: Avoid hardcoding parameters, paths, or magic numbers deep in the implementation code.
    * **Centralization**: Use centralized configuration mechanisms (e.g., config classes/files).

## 8. VERIFICATION PROTOCOL (STRICT)
* **No Assumptions**: NEVER assume the existence of files, versions, API keys, or specific configurations. Always establish ground truth first.
* **Ephemeral Information**: Your training data regarding API models, libraries, and external world states is **OUTDATED**.
    * **Action**: You MUST use `search_web` to verify **ANY** information that might have changed since your training cutoff (e.g., API endpoints, available models, library versions) before writing code.
* **Verify First**: Before using a specific value, **VERIFY** it exists or is correct:
    * **Local State**: Use `view_file`, `find_by_name`, `run_command` (e.g., check file exists, check installed version).
    * **External State**: Use `search_web` (e.g., check latest API docs, available models).
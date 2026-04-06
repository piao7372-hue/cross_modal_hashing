# cross_modal_hashing_starter

## Recommended location
You can place this folder on your Desktop, but avoid Chinese characters and spaces in the full path if possible.
Recommended example on Windows:

C:\Users\YourName\Desktop\cross_modal_hashing

## Suggested data layout
Put your raw datasets under `data/raw/` like this:

- data/raw/mirflickr25k/
- data/raw/nuswide/
- data/raw/mscoco/

Keep processed outputs under:

- data/processed/
- outputs/

## Local cache and temp directories
`.cache/` and `.tmp_pip/` are local runtime cache/temp directories. They are not source-of-truth for project status. Avoid keeping Hugging Face or pip temporary caches in the repo workspace long-term. For current project state and boundaries, use `docs/project_status.md`, `docs/semantic_cache_contract.md`, and `AGENTS.md`.

## First task for Codex
Ask Codex to inspect the repository and propose a plan before writing code.

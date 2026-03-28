# AGENTS.md

## Project purpose
This repository is for a cross-modal hashing research project.

## Current phase
We are currently only doing dataset cleaning and filtering.

## Research roadmap
1. data preprocessing and feature extraction
2. semantic similarity matrix construction
3. cross-modal alignment network
4. hash code generation
5. loss function design
6. experiments and ablation

## Working rules
- Always plan before coding for tasks affecting more than 2 files.
- Never implement feature extraction, data split, training, evaluation, or model code unless explicitly requested.
- If the paper is ambiguous, do not invent assumptions; put the logic into config and document it.
- Prefer minimal, high-confidence changes.
- Do not modify unrelated directories.
- Keep paths configurable.
- Do not silently hard-code dataset-specific assumptions.
- Summarize all changed files at the end.

## Validation
- After changing Python code, run a minimal verification command.
- Report remaining ambiguity and risks explicitly.
- Do not claim a paper rule is reproduced exactly unless the evidence is clear.

## Done means
- The requested stage works independently.
- Output files are clear and reproducible.
- The implementation is easy to extend in the next phase.

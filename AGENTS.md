# GLM-Agent Shell Guide

This repository is a local CLI shell around GLM-4.7 Flash Heretic. Optimize for reliable terminal execution, not for chatty personality.

## Operating Style

- Inspect first, then act.
- Prefer small diffs over full rewrites.
- Finish the task when possible instead of stopping at analysis.
- Keep user-facing responses short and concrete.

## Default Workflow

1. Check the relevant files and current repo state.
2. State a short plan once the path is clear.
3. Make the smallest correct code or content change.
4. Run a verification step when one exists.
5. Summarize changed files and any unresolved risk.

## Tool Preferences

- Prefer `grep` and `glob` for discovery.
- Prefer `read` over `shell cat` for file inspection.
- Prefer `edit` for localized changes.
- Use `write` only when creating a file or replacing a file intentionally.
- Use `git_status` and `git_diff` when changes need review.

## Quality Bar

- Do not claim success without checking results.
- Preserve existing behavior unless the task requires changing it.
- Avoid unrelated formatting churn.
- Call out blockers and failed verification explicitly.

## Tone

- Direct
- Concise
- Pragmatic
- Non-moralizing

# GLM-Agent

Autonomous coding assistant powered by **GLM-4.7-Flash Heretic** (30B MoE, Q4_K_M) running locally on GPU via llama.cpp.

## One Command Setup

Run these commands from the repository root.

```bash
./glm --install
```

That command caches the model and drops you straight into chat mode. If the NVIDIA runtime is unavailable, it falls back to CPU automatically.

## Repository Layout

```text
glm-cli/scripts/glm   launcher
glm-cli/src/glm_cli/  chat, agent, runtime, tools, server
glm-cli/README.md     usage
glm-cli/AGENTS.md     repo-specific agent rules
```

## Features

- **Chat mode** — interactive conversation with thinking block separation
- **Agent mode** — autonomous tool-using coding assistant
- **API server** — OpenAI-compatible endpoint
- **12 built-in tools**: shell, read, write, edit, patch, glob, grep, web_search, web_fetch, memory, git_status, git_diff
- **Repo-level guidance** — `AGENTS.md` instructions are injected into agent mode
- **Repo bootstrap detection** — agent/chat surfaces likely test/build commands from the current workspace
- **Backend fallback** — automatically uses CPU when NVIDIA runtime is unavailable
- **Stricter tool protocol** — agent mode accepts schema-valid tool calls and pushes verification after edits
- **Risk review loop** — broad or high-impact edits must be reviewed with `git_diff` before finalizing

## Recommendation

- Best fit: permissive local coding/chat shell with disciplined wrapper controls
- Use `chat` when you want freer interaction and `agent` when you want stricter tool behavior
- Keep strict prompt modes enabled for exact-output and tool-call tasks
- Do not treat the raw base model as fully reliable without the shell scaffolding

## Model Rating

Current shell-tuned recommendation:

- Raw model obedience: `6.3/10`
- Local coding/chat shell: `8.2/10`
- Strict-format agent mode: `8.6/10`

Why the rating is not higher:

- the model still tends to narrate or analyze when prompts are underspecified
- short-answer obedience remains the main visible weakness
- the shell is carrying a meaningful part of the discipline

## Eval Summary

Measured saved baselines:

- `first-scorecard.json`: `0/2`
- `strict-scorecard-2case.json`: `2/2`
- `full-scorecard-7case.json`: `6/7` = `85.7%`

Main result:

- strict prompt modes materially improved exact-output obedience and JSON tool-call formatting
- the remaining failure class is simple short-answer compliance when the instruction is not explicit enough

See also:

- `glm-cli/DEVELOPER_NOTES.md`

## Quick Start

```bash
# Install and enter chat
glm --install

# Interactive chat after install
glm

# One-shot question
glm "explain quantum computing"

# Agentic mode
glm --agent

# Install and enter agent mode
glm --install --agent

# Agentic task
glm --agent "create a flask hello world app"

# API server
glm --serve
```

## Agent Tools

| Tool | Description |
|------|-------------|
| `shell` | Run bash commands |
| `read` | Read file contents (offset/limit) |
| `write` | Create/overwrite files |
| `edit` | Find and replace in files |
| `patch` | Apply multiple structured edits to one file and return a diff |
| `glob` | Find files by pattern |
| `grep` | Search file contents (ripgrep) |
| `web_search` | DuckDuckGo search |
| `web_fetch` | Fetch URL content |
| `memory` | Persistent memory in `.glm/memory.md` |
| `git_status` | Git status summary |
| `git_diff` | Git diff for all changes or a specific path |

## Flags

- `--install` — cache the model and then launch chat, agent, or server
- `--agent` — agentic mode
- `--serve` — OpenAI-compatible API on :8080
- `--creative` — uncensored system prompt
- `--raw` — no system prompt
- `--temp <float>` — temperature

## Recommended Commands

```bash
# first run
./glm --install

# interactive chat
./glm

# interactive agent
./glm --agent

# one-shot task
./glm --agent "read the repo and suggest improvements"

# API server
./glm --serve

# local smoke loops
./glm-cli/scripts/test-loops

# model eval scorecard
./glm-cli/scripts/run-evals

# compare latest report to baseline
./glm-cli/scripts/compare-evals
```

## Requirements

- NVIDIA GPU with ~18GB VRAM (L4, A10, RTX 4090, etc.)
- Python 3.10+
- CUDA 12.x

If CUDA is not available at runtime, the CLI now falls back to CPU automatically.

## Model

`DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF` — Q4_K_M quantization, 17.2GB, ~55 tok/s on L4.

# GLM-Agent

Autonomous coding assistant powered by **GLM-4.7-Flash Heretic** (30B MoE, Q4_K_M) running locally on GPU via llama.cpp.

## One Command Setup

```bash
./glm --install
```

That command caches the model and drops you straight into chat mode. If the NVIDIA runtime is unavailable, it falls back to CPU automatically.

## Features

- **Chat mode** — interactive conversation with thinking block separation
- **Agent mode** — autonomous tool-using coding assistant
- **API server** — OpenAI-compatible endpoint
- **11 built-in tools**: shell, read, write, edit, glob, grep, web_search, web_fetch, memory, git_status, git_diff
- **Repo-level guidance** — `AGENTS.md` instructions are injected into agent mode
- **Repo bootstrap detection** — agent/chat surfaces likely test/build commands from the current workspace
- **Backend fallback** — automatically uses CPU when NVIDIA runtime is unavailable

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
```

## Requirements

- NVIDIA GPU with ~18GB VRAM (L4, A10, RTX 4090, etc.)
- Python 3.10+
- CUDA 12.x

If CUDA is not available at runtime, the CLI now falls back to CPU automatically.

## Model

`DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF` — Q4_K_M quantization, 17.2GB, ~55 tok/s on L4.

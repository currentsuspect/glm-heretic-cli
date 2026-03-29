# GLM-Agent

Autonomous coding assistant powered by **GLM-4.7-Flash Heretic** (30B MoE, Q4_K_M) running locally on GPU via llama.cpp.

## Features

- **Chat mode** — interactive conversation with thinking block separation
- **Agent mode** — autonomous tool-using coding assistant
- **API server** — OpenAI-compatible endpoint
- **9 built-in tools**: shell, read, write, edit, glob, grep, web_search, web_fetch, memory

## Quick Start

```bash
# Interactive chat
glm

# One-shot question
glm "explain quantum computing"

# Agentic mode
glm --agent

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

## Flags

- `--agent` — agentic mode
- `--serve` — OpenAI-compatible API on :8080
- `--creative` — uncensored system prompt
- `--raw` — no system prompt
- `--temp <float>` — temperature

## Requirements

- NVIDIA GPU with ~18GB VRAM (L4, A10, RTX 4090, etc.)
- Python 3.10+
- CUDA 12.x

## Model

`DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF` — Q4_K_M quantization, 17.2GB, ~55 tok/s on L4.

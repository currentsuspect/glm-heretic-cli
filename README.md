# glm-heretic-cli

The runnable project now lives in `glm-cli/`.

Quick start:

```bash
./glm --install
```

Keep using the same root commands:

```bash
./glm
./glm --agent
./glm --serve
```

Current recommendation:

- Best fit: local permissive coding/chat shell with strict wrapper controls enabled
- Current shell-tuned rating: `8.2/10`
- Main weakness: short-answer obedience can still drift into analysis on some prompts

Developer docs:

- `glm-cli/README.md`
- `glm-cli/DEVELOPER_NOTES.md`

Internal layout:

```text
glm-cli/
  AGENTS.md
  README.md
  scripts/glm
  src/glm_cli/
```

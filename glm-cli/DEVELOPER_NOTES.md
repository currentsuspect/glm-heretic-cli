# Developer Notes

## Model Snapshot

Base model:

- `DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF`

Shell status:

- permissive and usable in local CLI form
- meaningfully improved by strict wrapper modes
- not frontier-level raw instruction obedience without shell constraints

## Measured Eval State

Saved baselines:

- `evals/baselines/first-scorecard.json`: `0/2` on strict obedience and tool formatting
- `evals/baselines/strict-scorecard-2case.json`: `2/2` after strict prompt-mode tuning
- `evals/baselines/full-scorecard-7case.json`: `6/7` = `85.7%`

Observed improvement:

- exact-output obedience improved from failure to pass
- strict JSON tool-call formatting improved from failure to pass
- broader behavior is solid enough for local agent use, but still inconsistent on some simple short-answer prompts

## Rating

Recommended rating scale for the current shell-wrapped system:

- Raw model obedience: `6.3/10`
- Shell-tuned local coding/chat use: `8.2/10`
- Strict-format agent use: `8.6/10`

Interpretation:

- the base model is permissive and capable, but naturally verbose and prone to narration
- the wrapper is doing real work to make it feel disciplined
- this is a strong local agent shell, not a proven frontier coding model

## Recommendation

Recommended use:

- local coding assistant in repos where permissiveness matters
- CLI agent workflows that benefit from strict tool-call scaffolding
- chat + agent hybrid use where the shell can switch between expressive and strict modes

Not recommended as-is for:

- production automation that depends on perfect strict-format obedience
- unattended long-horizon refactors without human review
- benchmark claims against top closed coding agents without broader eval coverage

## Model Behavior Notes

Strengths:

- follows exact-output constraints much better under strict mode
- can emit valid schema-shaped tool calls when strongly instructed
- benefits from explicit inspect -> edit -> verify workflow framing
- works well with review gates and verification loops

Weaknesses:

- tends to over-explain if prompt contracts are loose
- can narrate instead of answering directly on short factual prompts
- needs wrapper-level enforcement for reliable tool use
- raw chat mode is noticeably less disciplined than agent mode

## Shell Decisions That Help This Model

Keep these features enabled:

- strict prompt modes for exact-output and tool-call tasks
- schema validation for tool parsing
- structured `patch` tool instead of broad rewrites
- verification retry loop after failed checks
- risky-edit review gating with `git_diff`
- repo bootstrap command detection

These are compensating controls for real model tendencies, not optional polish.

## Practical Recommendation

If the goal is a very good local agentic CLI:

- keep chat mode expressive
- keep agent mode rigid
- keep evals close to real tasks
- tune the wrapper against observed failures, not vibes

Best next tuning targets:

- improve short-answer obedience without requiring exact-output phrasing
- add more evals for multi-step coding tasks
- score edit quality and verification discipline, not only output formatting
- compare prompt variants on the same task pack before changing tools again

## Commands

Install and run:

```bash
./glm --install
./glm
./glm --agent
./glm --serve
```

Regression and evals:

```bash
./glm-cli/scripts/test-loops
./glm-cli/scripts/run-evals
./glm-cli/scripts/compare-evals
```

import sys
import os
import re
import json
import time
from huggingface_hub import hf_hub_download
from .tools import TOOLS, execute_tool
from .runtime import load_llm
from .context import (
    count_messages_tokens, compact_messages, get_git_context,
    save_session, load_session, list_sessions, MAX_CONTEXT_TOKENS, WARN_TOKENS,
    detect_repo_context, format_repo_context,
)

REPO = "DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF"
FILE = "GLM-4.7-Flash-Uncen-Hrt-NEO-CODE-MAX-imat-D_AU-Q4_K_M.gguf"

R       = "\033[0m"
B       = "\033[1m"
D       = "\033[2m"
I       = "\033[3m"
GRAY    = "\033[90m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
CYAN    = "\033[36m"
MAGENTA = "\033[35m"
WHITE   = "\033[97m"
BLUE    = "\033[34m"

MAX_TURNS = 20
MAX_TOKENS = 4096
SESSION_ID = None


def tool_required_fields():
    required = {}
    for name, spec in TOOLS.items():
        params = spec["schema"].get("parameters", {})
        required[name] = set(params.get("required", []))
    return required


TOOL_REQUIRED_FIELDS = tool_required_fields()


def clear_line():
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def spinner(msg="Loading"):
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    idx = 0
    while True:
        yield f"  {CYAN}{frames[idx % len(frames)]}{R} {GRAY}{msg}{R}"
        idx += 1


def render_panel(title, lines, accent=CYAN):
    width = 64
    title_text = f" {title} "
    rule = "─" * max(0, width - len(title_text))
    print(f"  {accent}{B}{title_text}{rule}{R}")
    for line in lines:
        print(f"  {GRAY}  {line}{R}")
    print()


def build_execution_plan(user_msg, cwd):
    repo = detect_repo_context(cwd)
    verify_cmd = None
    for item in repo["commands"]:
        if item["label"] == "test":
            verify_cmd = item["command"]
            break
    if verify_cmd is None and repo["commands"]:
        verify_cmd = repo["commands"][0]["command"]

    lowered = user_msg.lower()
    change_words = ("fix", "change", "edit", "update", "create", "implement", "refactor", "add")
    needs_change = any(word in lowered for word in change_words)

    steps = [
        "Inspect repo state and locate the relevant files.",
        "Read the target code before making changes.",
    ]
    if needs_change:
        steps.append("Make the smallest correct change that solves the task.")
    else:
        steps.append("Gather evidence and answer directly if no change is needed.")
    if verify_cmd:
        steps.append(f"Verify the result with `{verify_cmd}` or a targeted check.")
    else:
        steps.append("Review the resulting diff or run a targeted shell check before finishing.")
    return steps


def show_plan_panel(user_msg, cwd):
    steps = build_execution_plan(user_msg, cwd)
    lines = [f"task: {user_msg}"]
    lines.extend(f"{i + 1}. {step}" for i, step in enumerate(steps))
    render_panel("Plan", lines, accent=YELLOW)


def create_execution_state():
    return {
        "made_changes": False,
        "verified": False,
        "verification_prompted": False,
    }


def verification_hint(cwd):
    repo = detect_repo_context(cwd)
    for item in repo["commands"]:
        if item["label"] == "test":
            return f"Run `{item['command']}` before finalizing."
    if repo["commands"]:
        return f"Run `{repo['commands'][0]['command']}` or another targeted check before finalizing."
    return "Run a relevant shell check or inspect the diff before finalizing."


def update_execution_state(state, name, args, result):
    if name in {"edit", "write", "patch"}:
        state["made_changes"] = True
        return

    if name in {"git_diff", "git_status"}:
        return

    if name != "shell":
        return

    cmd = args.get("command", "").lower()
    if any(token in cmd for token in [
        "pytest", "npm test", "ruff check", "make test", "cargo test",
        "go test", "npm run lint", "npm run build", "cargo build", "go build",
    ]):
        if "[exit code:" not in result:
            state["verified"] = True

    if any(token in cmd for token in ["mkdir", "touch", "cp ", "mv ", "npm install", "pip install"]):
        state["made_changes"] = True


def load_model():
    sys.stderr.write(f"\n  {B}{CYAN}GLM-Agent{R} {D}· {len(TOOLS)} tools{R}\n")
    sys.stderr.write(f"  {D}30B MoE · 16K ctx · auto-compact · git-aware · repo-bootstrap{R}\n\n")
    sys.stderr.flush()

    spin = spinner("Resolving model...")
    sys.stderr.write(next(spin) + "\r")
    sys.stderr.flush()
    model_path = hf_hub_download(repo_id=REPO, filename=FILE)
    clear_line()
    sys.stderr.write(f"  {GREEN}✓{R} {GRAY}Model cached{R}\n")

    spin = spinner("Loading backend...")
    sys.stderr.write(next(spin) + "\r")
    sys.stderr.flush()
    try:
        llm, backend_note = load_llm(model_path=model_path, n_ctx=16384, verbose=False)
    except Exception as e:
        clear_line()
        sys.stderr.write(f"  {RED}✗{R} {GRAY}{e}{R}\n\n")
        sys.stderr.flush()
        raise SystemExit(1)
    clear_line()
    sys.stderr.write(f"  {GREEN}✓{R} {GRAY}{backend_note}{R}\n")

    spin = spinner("Warmup...")
    sys.stderr.write(next(spin) + "\r")
    sys.stderr.flush()
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4, temperature=0.1,
    )
    clear_line()
    sys.stderr.write(f"  {GREEN}✓{R} {GRAY}Ready{R}\n\n")
    sys.stderr.flush()
    return llm


def extract_json_objects(text):
    objects = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        if text[idx] != "{":
            idx += 1
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        idx += end
    return objects


def validate_tool_call(obj):
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    arguments = obj.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    if name not in TOOLS:
        return None
    missing = TOOL_REQUIRED_FIELDS.get(name, set()) - set(arguments.keys())
    if missing:
        return None
    return {"name": name, "arguments": arguments}


def parse_tool_calls(text):
    calls = []
    seen = set()

    fenced_matches = re.findall(r"```json\s*([\s\S]*?)```", text, re.DOTALL)
    for block in fenced_matches:
        for obj in extract_json_objects(block):
            call = validate_tool_call(obj)
            if call is None:
                continue
            key = json.dumps(call, sort_keys=True)
            if key not in seen:
                seen.add(key)
                calls.append(call)

    if calls:
        return calls

    for obj in extract_json_objects(text):
        call = validate_tool_call(obj)
        if call is None:
            continue
        key = json.dumps(call, sort_keys=True)
        if key not in seen:
            seen.add(key)
            calls.append(call)
    return calls


def render_thinking(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if not m:
        return
    thinking = m.group(1).strip()
    if not thinking:
        return
    print(f"  {YELLOW}{B}  thought{R}")
    print(f"  {GRAY}  {'─' * 52}{R}")
    for line in thinking.split("\n"):
        line = line.rstrip()
        if not line.strip():
            continue
        while len(line) > 76:
            print(f"  {GRAY}{I}  {line[:76]}{R}")
            line = line[76:]
        print(f"  {GRAY}{I}  {line}{R}")
    print(f"  {GRAY}  {'─' * 52}{R}")
    print()


def clean_answer(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```json\s*[\s\S]*?```', '', text, flags=re.DOTALL)
    for call in parse_tool_calls(text):
        text = text.replace(json.dumps(call), "")
    return text.strip()


def render_answer(text):
    text = clean_answer(text)
    if not text:
        return
    parts = re.split(r'(```[\s\S]*?```)', text)
    for part in parts:
        if part.startswith("```"):
            lines = part.split("\n")
            lang = lines[0][3:].strip()
            code = "\n".join(lines[1:])
            if code.endswith("```"):
                code = code[:-3]
            print(f"  {D}┌── {lang or 'code'} {'─' * (48 - len(lang))}{R}")
            for cl in code.split("\n"):
                print(f"  {D}│{R}  {WHITE}{cl}{R}")
            print(f"  {D}└{'─' * 54}{R}")
        else:
            for line in part.split("\n"):
                if line.strip():
                    print(f"  {line}")
    print()


def chat_once(llm, messages, temperature=0.7):
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=temperature,
        top_p=0.9,
        stop=["<|endoftext|>"],
    )
    return output["choices"][0]["message"]["content"], output["usage"]


def show_context_bar(messages):
    tokens = count_messages_tokens(messages)
    pct = int(tokens / MAX_CONTEXT_TOKENS * 100)
    bar_len = 20
    filled = int(pct / 100 * bar_len)
    if pct > 85:
        color = RED
    elif pct > 65:
        color = YELLOW
    else:
        color = GREEN
    bar = f"{color}{'█' * filled}{GRAY}{'░' * (bar_len - filled)}{R}"
    return f"  {GRAY}ctx {bar} {tokens}/{MAX_CONTEXT_TOKENS} ({pct}%){R}"


def show_repo_panel(cwd):
    repo = detect_repo_context(cwd)
    lines = []
    if repo["facts"]:
        lines.extend(repo["facts"][:6])
    if repo["commands"]:
        lines.append("Suggested: " + " | ".join(
            f"{item['label']}={item['command']}" for item in repo["commands"][:4]
        ))
    if not lines:
        lines.append("No repo bootstrap hints detected")
    render_panel("Workspace", [cwd] + lines, accent=BLUE)


def compact_if_needed(messages):
    tokens = count_messages_tokens(messages)
    if tokens < WARN_TOKENS:
        return messages
    print(f"  {YELLOW}⚠ Context near limit ({tokens}/{MAX_CONTEXT_TOKENS}) — compacting...{R}")
    messages, new_tokens = compact_messages(messages, MAX_CONTEXT_TOKENS)
    saved = tokens - new_tokens
    print(f"  {GRAY}  Compacted: {tokens} → {new_tokens} tokens (saved {saved}){R}\n")
    return messages


def auto_save(messages):
    global SESSION_ID
    cwd = os.getcwd()
    SESSION_ID, path = save_session(messages, cwd, SESSION_ID)


def agent_loop(llm, user_msg, messages, temperature=0.7):
    state = create_execution_state()
    show_plan_panel(user_msg, os.getcwd())
    messages.append({"role": "user", "content": user_msg})
    messages.append({
        "role": "system",
        "content": "Execution plan:\n" + "\n".join(
            f"{i + 1}. {step}" for i, step in enumerate(build_execution_plan(user_msg, os.getcwd()))
        ),
    })

    for turn in range(MAX_TURNS):
        # compact if needed
        messages = compact_if_needed(messages)

        # inject fresh git context on each turn
        git_ctx = get_git_context(os.getcwd())
        if git_ctx:
            messages.append({"role": "system", "content": f"Git context:\n{git_ctx}"})

        # show context bar
        print(show_context_bar(messages))

        start = time.time()
        resp, usage = chat_once(llm, messages, temperature)
        elapsed = time.time() - start

        # remove injected git context
        if git_ctx:
            messages.pop()

        # show thinking
        render_thinking(resp)

        # check for tool calls
        calls = parse_tool_calls(resp)

        if not calls:
            if state["made_changes"] and not state["verified"] and not state["verification_prompted"]:
                render_panel("Verify", [verification_hint(os.getcwd())], accent=RED)
                messages.append({"role": "assistant", "content": resp})
                messages.append({
                    "role": "user",
                    "content": "You have made changes but have not verified them yet. Run a relevant verification step before giving the final answer.",
                })
                state["verification_prompted"] = True
                continue
            render_answer(resp)
            messages.append({"role": "assistant", "content": resp})
            print(f"  {GRAY}└ {usage['completion_tokens']} tok · {elapsed:.1f}s · {usage['completion_tokens']/elapsed:.0f} tok/s{R}\n")
            auto_save(messages)
            return

        render_answer(resp)
        messages.append({"role": "assistant", "content": resp})

        for call in calls:
            name = call["name"]
            args = call.get("arguments", {})
            args_str = json.dumps(args)
            if len(args_str) > 70:
                args_str = args_str[:67] + "..."

            print(f"  {MAGENTA}{B}⚡{R} {B}{name}{R} {GRAY}{args_str}{R}")

            result = execute_tool(name, args)
            result = truncate(result, 4000)

            print(f"  {D}{'─' * 56}{R}")
            for line in result.split("\n"):
                print(f"  {GRAY}│{R} {line}")
            print(f"  {D}{'─' * 56}{R}\n")

            update_execution_state(state, name, args, result)

            messages.append({
                "role": "user",
                "content": f"Tool result for `{name}`:\n{result}"
            })

    print(f"  {RED}⚠ Max turns ({MAX_TURNS}) reached{R}\n")
    auto_save(messages)


def truncate(text, limit=4000):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... ({len(text) - limit} chars truncated)"


def build_system_prompt():
    tools_desc = ""
    for name, t in TOOLS.items():
        schema = json.dumps(t["schema"], indent=2)
        tools_desc += f"### {name}\n```json\n{schema}\n```\n\n"

    header = """You are GLM-Agent, a pragmatic terminal coding agent.

Your job is to complete the user's task end-to-end using the available tools. The model may be permissive; your behavior must still be disciplined, accurate, and action-oriented.

## Core Priorities

1. Solve the user's actual request.
2. Inspect the workspace before editing.
3. Make the smallest correct change.
4. Preserve existing user work.
5. Verify results before claiming success.
6. Keep responses concise and operational.

## Execution Loop

1. Briefly say what you will inspect or do.
2. Explore relevant files or commands.
3. Form a short plan once the path is clear.
4. Execute changes with tools.
5. Verify with shell commands or targeted checks.
6. End with a clear final answer and no tool calls.

## Tool Call Format

To call a tool, output a fenced JSON block like this:

```json
{"name": "tool_name", "arguments": {"key": "value"}}
```

Only emit valid tool names and arguments that match the tool schema. You may call multiple tools in sequence. Prefer acting over discussing when tools can answer the question.

## Tool Discipline

- Read before write unless the task is a trivial new file.
- Prefer `glob` and `grep` for discovery.
- Use `read` with `offset` and `limit` for large files.
- Use `edit` for targeted replacements.
- Use `patch` for multi-step edits to one file when one replacement is not enough.
- Use `write` for intentional file creation or full replacement.
- Use `shell` for inspection, tests, formatting, and builds.
- Use `git_status` before risky edits when the repo may already be dirty.
- Use `git_diff` after edits to inspect the actual change.
- Use `web_search` and `web_fetch` only when local context is insufficient.
- Use `memory` only for facts worth carrying across sessions.

## Editing Rules

- Do not overwrite unrelated changes.
- Do not perform broad rewrites unless requested.
- Keep diffs focused and readable.
- If a command or edit fails, explain it briefly and try the next reasonable step.
- If verification fails, continue working unless blocked.
- If you make changes, do not finalize until you have run a relevant verification step or clearly explained why verification is unavailable.

## Communication Rules

- Be concise.
- Avoid filler and motivational language.
- Prefer concrete statements over general advice.
- Keep reasoning short before tool use.

## Safety Boundaries

- Do not perform destructive workspace operations unless the user explicitly requests them.
- Do not fabricate command output, file contents, or test results.
- Inspect or verify instead of guessing.

## Available Tools

"""

    rules = f"\n## Working Directory\n\n{os.getcwd()}\n"

    repo_ctx = format_repo_context(detect_repo_context(os.getcwd()))
    if repo_ctx:
        rules += f"\n## Repository Bootstrap\n{repo_ctx}\n"

    # inject git context at start
    git_ctx = get_git_context(os.getcwd())
    if git_ctx:
        rules += f"\n## Current Git State\n{git_ctx}\n"

    agents_path = os.path.join(os.getcwd(), "AGENTS.md")
    if os.path.exists(agents_path):
        try:
            with open(agents_path, "r", encoding="utf-8", errors="replace") as f:
                agents_text = f.read().strip()
            if agents_text:
                rules += f"\n## Repository Instructions\n\n{agents_text}\n"
        except OSError:
            pass

    return header + tools_desc + rules


def main():
    global SESSION_ID
    flags = sys.argv[1:]
    prompt_args = []
    temperature = 0.7

    i = 0
    while i < len(flags):
        if flags[i] == "--temp" and i + 1 < len(flags):
            temperature = float(flags[i + 1])
            i += 1
        elif flags[i] == "--resume":
            # try to resume last session
            cwd = os.getcwd()
            msgs, sid = load_session(cwd)
            if msgs:
                print(f"  {GRAY}Resumed session {sid} ({len(msgs)} messages){R}\n")
                SESSION_ID = sid
                llm = load_model()
                agent_loop(llm, "Continue where we left off.", msgs, temperature)
                return
            else:
                print(f"  {RED}No sessions found{R}")
                return
        elif flags[i] == "--sessions":
            cwd = os.getcwd()
            sessions = list_sessions(cwd)
            if not sessions:
                print(f"  {GRAY}No saved sessions{R}")
            else:
                print(f"\n  {B}{CYAN}Sessions:{R}\n")
                for s in sessions[:10]:
                    print(f"  {GREEN}{s['id']}{R}  {GRAY}{s['time']} · {s['messages']} msgs{R}")
                    if s["preview"]:
                        print(f"  {GRAY}  {s['preview'][:60]}{R}")
                print()
            return
        elif flags[i] == "--help":
            print(f"""
  {B}{CYAN}GLM-Agent{R} — autonomous coding assistant

  {B}Usage:{R}
    glm --agent                     interactive agent
    glm --agent "task"              one-shot task
    glm --agent --resume            resume last session
    glm --agent --sessions          list saved sessions
    glm --agent --temp 0.3 "task"   set temperature

  {B}Interactive:{R}
    quit, exit, q                   exit
    reset                           clear history
    /system <text>                  append to system prompt
    /tools                          list available tools
    /repo                           show detected repo info
    /ctx                            show context usage
    /sessions                       list saved sessions
""")
            return
        else:
            prompt_args.append(flags[i])
        i += 1

    one_shot = " ".join(prompt_args) if prompt_args else None
    llm = load_model()

    system = build_system_prompt()

    # load memory if exists
    mem_path = os.path.join(os.getcwd(), ".glm", "memory.md")
    if os.path.exists(mem_path):
        with open(mem_path) as f:
            mem = f.read().strip()
        if mem:
            system += f"\n\n## Saved Memory\n{mem}"

    messages = [{"role": "system", "content": system}]

    if one_shot:
        agent_loop(llm, one_shot, messages, temperature)
        return

    # interactive
    cwd = os.getcwd()
    render_panel(
        "GLM-Agent",
        [
            f"{len(TOOLS)} tools | auto-compact | git-aware | repo-bootstrap",
            "Ctrl+C exits and auto-saves to .glm/sessions/",
            "Use /help for commands",
        ],
    )
    show_repo_panel(cwd)

    while True:
        try:
            user = input(f"  {CYAN}{B}?{R}  ").strip()
        except (EOFError, KeyboardInterrupt):
            auto_save(messages)
            print(f"\n  {GRAY}Session saved. Bye.{R}")
            break

        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            auto_save(messages)
            print(f"  {GRAY}Session saved. Bye.{R}")
            break
        if user.lower() == "reset":
            messages = [{"role": "system", "content": system}]
            print(f"  {GRAY}History cleared.{R}\n")
            continue
        if user.startswith("/system "):
            system += "\n" + user[8:]
            messages[0] = {"role": "system", "content": system}
            print(f"  {GRAY}System updated.{R}\n")
            continue
        if user == "/tools":
            print()
            for name, t in TOOLS.items():
                desc = t["schema"]["description"]
                print(f"  {MAGENTA}{B}{name:14}{R} {GRAY}{desc}{R}")
            print()
            continue
        if user == "/repo":
            show_repo_panel(cwd)
            continue
        if user == "/ctx":
            print(show_context_bar(messages) + "\n")
            continue
        if user == "/sessions":
            sessions = list_sessions(cwd)
            if not sessions:
                print(f"  {GRAY}No saved sessions{R}\n")
            else:
                print()
                for s in sessions[:5]:
                    print(f"  {GREEN}{s['id']}{R}  {GRAY}{s['time']} · {s['messages']} msgs{R}")
                print()
            continue
        if user == "/help":
            print(f"""
  {B}Commands:{R}
    {B}/system <text>{R}   append to system prompt
    {B}/tools{R}           list tools
    {B}/repo{R}            show detected repo info
    {B}/ctx{R}             show context usage
    {B}/sessions{R}        list saved sessions
    {B}reset{R}            clear history
    {B}quit{R}             exit (auto-saves)

  {B}Example tasks:{R}
    "list all python files and count lines"
    "create a flask hello world app"
    "check git status and show me what changed"
    "search for latest llama.cpp release"
    "read src/main.py and fix the bug"
    "remember: we use pytest for testing"
""")
            continue

        agent_loop(llm, user, messages, temperature)


if __name__ == "__main__":
    main()

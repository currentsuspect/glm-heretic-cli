import os
import json
import time
import subprocess
import difflib

SESSION_DIR = ".glm/sessions"
MAX_CONTEXT_TOKENS = 7000  # leave room for output in 8K ctx
WARN_TOKENS = 5500


def estimate_tokens(text):
    return len(text) // 3  # rough: 3 chars ≈ 1 token for mixed en/code


def count_messages_tokens(messages):
    total = 0
    for m in messages:
        total += estimate_tokens(m.get("content", "")) + 10  # overhead per message
    return total


def get_git_context(cwd):
    lines = []
    try:
        r = subprocess.run(
            "git status --short 2>/dev/null",
            shell=True, capture_output=True, text=True, cwd=cwd, timeout=5
        )
        if r.stdout.strip():
            lines.append("Git status:")
            lines.append(r.stdout.strip())
    except Exception:
        pass
    try:
        r = subprocess.run(
            "git diff --stat HEAD 2>/dev/null | tail -5",
            shell=True, capture_output=True, text=True, cwd=cwd, timeout=5
        )
        if r.stdout.strip():
            lines.append("Recent changes:")
            lines.append(r.stdout.strip())
    except Exception:
        pass
    return "\n".join(lines) if lines else ""


def detect_repo_context(cwd):
    facts = []
    commands = []

    def add_command(label, cmd):
        if not any(existing["label"] == label for existing in commands):
            commands.append({"label": label, "command": cmd})

    package_json = os.path.join(cwd, "package.json")
    pyproject = os.path.join(cwd, "pyproject.toml")
    requirements = os.path.join(cwd, "requirements.txt")
    cargo = os.path.join(cwd, "Cargo.toml")
    gomod = os.path.join(cwd, "go.mod")
    makefile = os.path.join(cwd, "Makefile")

    if os.path.exists(package_json):
        facts.append("Node project detected via package.json")
        try:
            with open(package_json, "r", encoding="utf-8") as f:
                pkg = json.load(f)
            scripts = pkg.get("scripts", {})
            if scripts:
                facts.append("npm scripts: " + ", ".join(sorted(scripts.keys())[:8]))
            if "test" in scripts:
                add_command("test", "npm test")
            if "lint" in scripts:
                add_command("lint", "npm run lint")
            if "build" in scripts:
                add_command("build", "npm run build")
            if "dev" in scripts:
                add_command("dev", "npm run dev")
            if "start" in scripts:
                add_command("start", "npm start")
        except Exception:
            facts.append("package.json present but could not be parsed")

    if os.path.exists(pyproject):
        facts.append("Python project detected via pyproject.toml")
        add_command("test", "pytest")
        add_command("lint", "ruff check .")
    elif os.path.exists(requirements):
        facts.append("Python project detected via requirements.txt")
        add_command("test", "pytest")

    if os.path.exists(cargo):
        facts.append("Rust project detected via Cargo.toml")
        add_command("test", "cargo test")
        add_command("build", "cargo build")

    if os.path.exists(gomod):
        facts.append("Go project detected via go.mod")
        add_command("test", "go test ./...")
        add_command("build", "go build ./...")

    if os.path.exists(makefile):
        facts.append("Makefile detected")
        add_command("build", "make")
        add_command("test", "make test")

    try:
        entries = sorted(name for name in os.listdir(cwd) if not name.startswith("."))[:12]
        if entries:
            facts.append("Top-level files: " + ", ".join(entries))
    except Exception:
        pass

    return {"facts": facts, "commands": commands}


def format_repo_context(repo):
    if not repo["facts"] and not repo["commands"]:
        return ""

    lines = []
    if repo["facts"]:
        lines.append("Repository facts:")
        for fact in repo["facts"]:
            lines.append(f"- {fact}")
    if repo["commands"]:
        lines.append("Suggested commands:")
        for item in repo["commands"]:
            lines.append(f"- {item['label']}: {item['command']}")
    return "\n".join(lines)


def compact_messages(messages, max_tokens=MAX_CONTEXT_TOKENS):
    current = count_messages_tokens(messages)
    if current < max_tokens:
        return messages, current

    # strategy: keep system prompt + first user msg + last N messages that fit
    system = messages[0] if messages and messages[0]["role"] == "system" else None
    rest = messages[1:] if system else messages[:]

    # find first user message (original task)
    first_user = None
    remaining = []
    for m in rest:
        if m["role"] == "user" and first_user is None:
            first_user = m
        else:
            remaining.append(m)

    # keep as many recent messages as fit
    budget = max_tokens - (estimate_tokens(system["content"]) if system else 0)
    budget -= estimate_tokens(first_user["content"]) if first_user else 0
    budget -= 200  # safety margin

    kept = []
    used = 0
    for m in reversed(remaining):
        cost = estimate_tokens(m["content"]) + 10
        if used + cost > budget:
            break
        kept.insert(0, m)
        used += cost

    # build summary of dropped messages
    dropped = [m for m in remaining if m not in kept]
    if dropped:
        summary_parts = []
        tool_calls = 0
        tool_results = 0
        for m in dropped:
            if "Tool result for" in m.get("content", ""):
                tool_results += 1
            elif m["role"] == "assistant":
                tool_calls += 1
        if tool_calls or tool_results:
            summary = f"[Context compacted: {len(dropped)} messages dropped ({tool_calls} responses, {tool_results} tool results)]"
            kept.insert(0, {"role": "system", "content": summary})

    result = []
    if system:
        result.append(system)
    if first_user:
        result.append(first_user)
    result.extend(kept)

    new_count = count_messages_tokens(result)
    return result, new_count


def save_session(messages, cwd, session_id=None):
    session_dir = os.path.join(cwd, SESSION_DIR)
    os.makedirs(session_dir, exist_ok=True)

    if session_id is None:
        session_id = time.strftime("%Y%m%d-%H%M%S")

    path = os.path.join(session_dir, f"{session_id}.json")
    data = {
        "id": session_id,
        "timestamp": time.time(),
        "messages": messages,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return session_id, path


def load_session(cwd, session_id=None):
    session_dir = os.path.join(cwd, SESSION_DIR)
    if not os.path.exists(session_dir):
        return None, None

    if session_id:
        path = os.path.join(session_dir, f"{session_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return data["messages"], session_id
        return None, None

    # load most recent
    sessions = sorted(os.listdir(session_dir), reverse=True)
    for s in sessions:
        if s.endswith(".json"):
            path = os.path.join(session_dir, s)
            with open(path) as f:
                data = json.load(f)
            return data["messages"], s.replace(".json", "")
    return None, None


def list_sessions(cwd):
    session_dir = os.path.join(cwd, SESSION_DIR)
    if not os.path.exists(session_dir):
        return []
    results = []
    for s in sorted(os.listdir(session_dir), reverse=True):
        if s.endswith(".json"):
            path = os.path.join(session_dir, s)
            with open(path) as f:
                data = json.load(f)
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(data["timestamp"]))
            msg_count = len(data["messages"])
            first_user = ""
            for m in data["messages"]:
                if m["role"] == "user":
                    first_user = m["content"][:60]
                    break
            results.append({
                "id": data["id"],
                "time": ts,
                "messages": msg_count,
                "preview": first_user,
            })
    return results

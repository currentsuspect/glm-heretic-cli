import json
import os
import re
import subprocess
import glob as globmod
import urllib.request
import urllib.parse
import shlex
import difflib

WORKDIR = os.getcwd()
MAX_OUTPUT = 4000
SAFE_MODE = True  # require confirmation for destructive ops


def truncate(text, limit=MAX_OUTPUT):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... ({len(text) - limit} chars truncated)"


def confirm(prompt):
    if not SAFE_MODE:
        return True
    try:
        resp = input(f"  {prompt} [y/N] ").strip().lower()
        return resp in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def classify_shell_command(cmd):
    lowered = cmd.lower()
    block_patterns = [
        "rm -rf /", "mkfs", "dd if=", "shutdown", "reboot",
        "poweroff", "halt", "git reset --hard", "git clean -fd",
        ":(){:|:&};:", "chmod -r 777 /", "chown -r /",
    ]
    caution_patterns = [
        "rm ", "mv ", "cp ", "curl ", "wget ", "pip install",
        "npm install", "npm i ", "pnpm add", "yarn add",
        "git checkout ", "git switch ", "git restore ",
        "docker ", "sudo ", "python -m pip",
    ]

    for pattern in block_patterns:
        if pattern in lowered:
            return "blocked", f"blocked destructive pattern: {pattern}"

    if re.search(r"\brm\s+-[^\n]*[rf][^\n]*\s+(\.|/|\.\.)", lowered):
        return "blocked", "blocked recursive delete command"
    if re.search(r"\bmv\s+.+\s+/dev/null\b", lowered):
        return "blocked", "blocked destructive move"
    if re.search(r"\bchmod\s+-r\b", lowered):
        return "caution", "recursive permission change"

    for pattern in caution_patterns:
        if pattern in lowered:
            return "caution", f"state-changing command: {pattern.strip()}"

    return "safe", "read-only or low-risk command"


def make_diff(path, old_content, new_content):
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}"))
    return "".join(diff)


# ANSI for tool display
R = "\033[0m"
B = "\033[1m"
D = "\033[2m"
GRAY = "\033[90m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
WHITE = "\033[97m"


# ── shell ──────────────────────────────────────────────────────────────

def tool_shell(args):
    cmd = args.get("command", "")
    if not cmd:
        return "Error: no command provided"
    timeout = min(args.get("timeout", 30), 120)
    confirm_dangerous = args.get("confirm_dangerous", False)
    risk, reason = classify_shell_command(cmd)

    if risk == "blocked":
        return (
            "[blocked]\n"
            f"Command was not run because it looks destructive: {reason}\n"
            "If the user explicitly wants this, rerun with confirm_dangerous=true."
        )

    if risk == "caution" and confirm_dangerous:
        if not confirm(f"Run caution command? {cmd}"):
            return "[cancelled]\nUser declined to run the caution command."

    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=WORKDIR,
        )
        out = f"[risk: {risk}] {reason}\n"
        if r.stdout:
            out += r.stdout
        if r.stderr:
            out += f"\n[stderr]\n{r.stderr}"
        if r.returncode != 0:
            out += f"\n[exit code: {r.returncode}]"
        return truncate(out.strip()) or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


# ── read ───────────────────────────────────────────────────────────────

def tool_read(args):
    path = args.get("path", "")
    offset = args.get("offset", 1)
    limit = args.get("limit", 200)
    if not path:
        return "Error: no path provided"
    full = os.path.join(WORKDIR, path) if not os.path.isabs(path) else path
    if not os.path.exists(full):
        return f"Error: file not found: {path}"
    try:
        with open(full, "r", errors="replace") as f:
            lines = f.readlines()
        start = max(0, offset - 1)
        end = min(len(lines), start + limit)
        result = ""
        for i in range(start, end):
            result += f"{i+1}: {lines[i]}"
        if end < len(lines):
            result += f"\n... ({len(lines) - end} more lines)"
        return truncate(result)
    except Exception as e:
        return f"Error: {e}"


# ── write ──────────────────────────────────────────────────────────────

def tool_write(args):
    path = args.get("path", "")
    content = args.get("content", "")
    if not path:
        return "Error: no path provided"
    full = os.path.join(WORKDIR, path) if not os.path.isabs(path) else path
    try:
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
        lines = content.count("\n") + 1
        return f"Wrote {lines} lines to {path}"
    except Exception as e:
        return f"Error: {e}"


# ── edit ───────────────────────────────────────────────────────────────

def tool_edit(args):
    path = args.get("path", "")
    old = args.get("old_string", "")
    new = args.get("new_string", "")
    if not path or not old:
        return "Error: path and old_string required"
    full = os.path.join(WORKDIR, path) if not os.path.isabs(path) else path
    if not os.path.exists(full):
        return f"Error: file not found: {path}"
    try:
        with open(full, "r") as f:
            content = f.read()
        count = content.count(old)
        if count == 0:
            return "Error: old_string not found in file"
        if count > 1:
            return f"Error: old_string found {count} times — provide more context to make it unique"
        content = content.replace(old, new)
        with open(full, "w") as f:
            f.write(content)
        old_lines = old.count("\n") + 1
        new_lines = new.count("\n") + 1
        return f"Replaced {old_lines} lines with {new_lines} lines in {path}"
    except Exception as e:
        return f"Error: {e}"


# ── glob ───────────────────────────────────────────────────────────────

def tool_glob(args):
    pattern = args.get("pattern", "")
    if not pattern:
        return "Error: no pattern provided"
    try:
        matches = globmod.glob(pattern, root_dir=WORKDIR, recursive=True)
        if not matches:
            return "No files matched"
        return "\n".join(sorted(matches)[:100])
    except Exception as e:
        return f"Error: {e}"


# ── grep ───────────────────────────────────────────────────────────────

def tool_grep(args):
    pattern = args.get("pattern", "")
    path = args.get("path", ".")
    include = args.get("include", None)
    if not pattern:
        return "Error: no pattern provided"
    full = os.path.join(WORKDIR, path) if not os.path.isabs(path) else path
    cmd = f"rg --line-number --max-count 50 {shlex.quote(pattern)} {shlex.quote(full)}"
    if include:
        cmd += f" --glob {shlex.quote(include)}"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        return truncate(r.stdout.strip()) or "No matches"
    except FileNotFoundError:
        # fallback to grep
        cmd = f"grep -rn --include='{include}' -m 50 {shlex.quote(pattern)} {shlex.quote(full)}" if include else \
              f"grep -rn -m 50 {shlex.quote(pattern)} {shlex.quote(full)}"
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
            return truncate(r.stdout.strip()) or "No matches"
        except Exception as e:
            return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


# ── web_search ─────────────────────────────────────────────────────────

def tool_web_search(args):
    query = args.get("query", "")
    if not query:
        return "Error: no query provided"
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        # extract result snippets
        results = []
        for m in re.finditer(r'class="result__snippet"[^>]*>(.*?)</a', html, re.S):
            text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            if text:
                results.append(text)
        for m in re.finditer(r'class="result__url"[^>]*>(.*?)</a', html, re.S):
            link = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            if link and len(results) > 0:
                results[0] = f"[{link}] {results[0]}"
        if not results:
            return "No results found"
        return "\n\n".join(f"{i+1}. {r}" for i, r in enumerate(results[:5]))
    except Exception as e:
        return f"Error: {e}"


# ── web_fetch ──────────────────────────────────────────────────────────

def tool_web_fetch(args):
    url = args.get("url", "")
    if not url:
        return "Error: no url provided"
    if not url.startswith("http"):
        url = f"https://{url}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        # basic html to text
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return truncate(text[:8000])
    except Exception as e:
        return f"Error: {e}"


# ── memory ─────────────────────────────────────────────────────────────

def tool_memory(args):
    action = args.get("action", "read")
    content = args.get("content", "")
    mem_dir = os.path.join(WORKDIR, ".glm")
    mem_file = os.path.join(mem_dir, "memory.md")
    os.makedirs(mem_dir, exist_ok=True)

    if action == "read":
        if os.path.exists(mem_file):
            with open(mem_file, "r") as f:
                return f.read() or "(empty)"
        return "(no memory file)"
    elif action == "write":
        with open(mem_file, "w") as f:
            f.write(content)
        return f"Memory written ({len(content)} chars)"
    elif action == "append":
        with open(mem_file, "a") as f:
            f.write("\n" + content)
        return f"Memory appended ({len(content)} chars)"
    else:
        return f"Error: unknown action '{action}' (use read/write/append)"


# ── git ────────────────────────────────────────────────────────────────

def tool_git_status(args):
    try:
        r = subprocess.run(
            ["git", "status", "--short", "--branch"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=WORKDIR,
        )
        out = (r.stdout or "") + (f"\n[stderr]\n{r.stderr}" if r.stderr else "")
        if r.returncode != 0:
            return truncate(out.strip() or "Error: git status failed")
        return truncate(out.strip() or "Working tree clean")
    except FileNotFoundError:
        return "Error: git is not installed"
    except Exception as e:
        return f"Error: {e}"


def tool_git_diff(args):
    path = args.get("path")
    staged = args.get("staged", False)

    cmd = ["git", "diff"]
    if staged:
        cmd.append("--staged")
    if path:
        cmd.extend(["--", path])

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=WORKDIR,
        )
        out = (r.stdout or "") + (f"\n[stderr]\n{r.stderr}" if r.stderr else "")
        if r.returncode != 0:
            return truncate(out.strip() or "Error: git diff failed")
        return truncate(out.strip() or "No diff")
    except FileNotFoundError:
        return "Error: git is not installed"
    except Exception as e:
        return f"Error: {e}"


# ── tool registry ──────────────────────────────────────────────────────

TOOLS = {
    "shell": {
        "fn": tool_shell,
        "schema": {
            "name": "shell",
            "description": "Run a shell command. Returns stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (max 120)", "default": 30},
                    "confirm_dangerous": {"type": "boolean", "description": "Ask for confirmation before running a caution command", "default": False},
                },
                "required": ["command"],
            },
        },
    },
    "read": {
        "fn": tool_read,
        "schema": {
            "name": "read",
            "description": "Read a file. Shows line numbers. Use offset/limit for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative or absolute)"},
                    "offset": {"type": "integer", "description": "Start line number (1-indexed)", "default": 1},
                    "limit": {"type": "integer", "description": "Max lines to read", "default": 200},
                },
                "required": ["path"],
            },
        },
    },
    "write": {
        "fn": tool_write,
        "schema": {
            "name": "write",
            "description": "Write content to a file. Creates parent dirs. Overwrites existing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    "edit": {
        "fn": tool_edit,
        "schema": {
            "name": "edit",
            "description": "Find and replace text in a file. old_string must be unique.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old_string": {"type": "string", "description": "Text to find"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    "glob": {
        "fn": tool_glob,
        "schema": {
            "name": "glob",
            "description": "Find files matching a glob pattern (e.g., '**/*.py')",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                },
                "required": ["pattern"],
            },
        },
    },
    "grep": {
        "fn": tool_grep,
        "schema": {
            "name": "grep",
            "description": "Search file contents with regex. Uses ripgrep if available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Search path", "default": "."},
                    "include": {"type": "string", "description": "File glob filter (e.g., '*.py')"},
                },
                "required": ["pattern"],
            },
        },
    },
    "web_search": {
        "fn": tool_web_search,
        "schema": {
            "name": "web_search",
            "description": "Search the web via DuckDuckGo. Returns top 5 results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    "web_fetch": {
        "fn": tool_web_fetch,
        "schema": {
            "name": "web_fetch",
            "description": "Fetch a URL and return text content (HTML stripped).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
    "memory": {
        "fn": tool_memory,
        "schema": {
            "name": "memory",
            "description": "Read/write/append to persistent memory file (.glm/memory.md).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read", "write", "append"], "description": "Action"},
                    "content": {"type": "string", "description": "Content for write/append"},
                },
                "required": ["action"],
            },
        },
    },
    "git_status": {
        "fn": tool_git_status,
        "schema": {
            "name": "git_status",
            "description": "Show git status (modified, added, deleted files).",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    "git_diff": {
        "fn": tool_git_diff,
        "schema": {
            "name": "git_diff",
            "description": "Show git diff for a file or all changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (omit for all changes)"},
                    "staged": {"type": "boolean", "description": "Show staged changes", "default": False},
                },
            },
        },
    },
}


def execute_tool(name, args):
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'. Available: {', '.join(TOOLS.keys())}"
    return TOOLS[name]["fn"](args)

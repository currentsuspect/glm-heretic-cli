import sys
import os
import re
import json
import time
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from glm_tools import TOOLS, execute_tool

REPO = "DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF"
FILE = "GLM-4.7-Flash-Uncen-Hrt-NEO-CODE-MAX-imat-D_AU-Q4_K_M.gguf"

R     = "\033[0m"
B     = "\033[1m"
D     = "\033[2m"
I     = "\033[3m"
GRAY  = "\033[90m"
RED   = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN  = "\033[36m"
MAGENTA = "\033[35m"
WHITE = "\033[97m"

MAX_TURNS = 20
MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are GLM-Agent, an autonomous coding assistant. You have access to tools.

## How to Use Tools

To call a tool, output a JSON block like this:

```json
{"name": "tool_name", "arguments": {"key": "value"}}
```

You can call multiple tools by outputting multiple JSON blocks. You can also provide reasoning text before tool calls. After tools execute, you'll get results and can continue.

## Available Tools

{tools}

## Rules

1. Think step by step. Show your reasoning before calling tools.
2. Explore before making changes - read files first, glob to find things.
3. Test your changes with shell commands after making them.
4. When done, give a clear final answer (no tool call needed).
5. For long files, use offset/limit parameters.
6. Use web_search for recent info, docs, or unknowns.
7. Use memory tool to save important context that persists across sessions.
8. Be concise in your reasoning. Be thorough in your work.

## Working Directory: {cwd}
"""


def clear_line():
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def load_model():
    sys.stderr.write(f"\n  {B}{CYAN}GLM-Agent{R} {D}agentic mode · {len(TOOLS)} tools{R}\n")
    sys.stderr.write(f"  {D}30B MoE · 16K agent ctx{R}\n\n")
    sys.stderr.flush()

    sys.stderr.write(f"  {GRAY}Downloading...{R}\r")
    sys.stderr.flush()
    model_path = hf_hub_download(repo_id=REPO, filename=FILE)
    clear_line()
    sys.stderr.write(f"  {GREEN}✓{R} {GRAY}Model cached{R}\n")

    sys.stderr.write(f"  {GRAY}Loading GPU...{R}\r")
    sys.stderr.flush()
    llm = Llama(model_path=model_path, n_ctx=16384, n_gpu_layers=-1, verbose=False)
    clear_line()
    sys.stderr.write(f"  {GREEN}✓{R} {GRAY}GPU ready{R}\n")

    sys.stderr.write(f"  {GRAY}Warmup...{R}\r")
    sys.stderr.flush()
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4, temperature=0.1,
    )
    clear_line()
    sys.stderr.write(f"  {GREEN}✓{R} {GRAY}Ready{R}\n\n")
    sys.stderr.flush()
    return llm


def parse_tool_calls(text):
    calls = []
    for m in re.finditer(r'```json\s*(\{[^`]*?\})\s*```', text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            if "name" in obj and "arguments" in obj:
                calls.append(obj)
        except json.JSONDecodeError:
            pass
    # also try raw JSON without code fences
    if not calls:
        for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"[^"]*"[^{}]*"arguments"\s*:\s*\{[^}]*\}[^{}]*\}', text):
            try:
                obj = json.loads(m.group(0))
                if "name" in obj and "arguments" in obj:
                    calls.append(obj)
            except json.JSONDecodeError:
                pass
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
    text = re.sub(r'```json\s*\{[^`]*?\}\s*```', '', text, flags=re.DOTALL)
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


def truncate(text, limit=4000):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... ({len(text) - limit} chars truncated)"


def agent_loop(llm, user_msg, messages, temperature=0.7):
    messages.append({"role": "user", "content": user_msg})

    for turn in range(MAX_TURNS):
        start = time.time()
        resp, usage = chat_once(llm, messages, temperature)
        elapsed = time.time() - start

        # show thinking
        render_thinking(resp)

        # check for tool calls
        calls = parse_tool_calls(resp)

        if not calls:
            # final answer
            render_answer(resp)
            messages.append({"role": "assistant", "content": resp})
            print(f"  {GRAY}└ {usage['completion_tokens']} tok · {elapsed:.1f}s · {usage['completion_tokens']/elapsed:.0f} tok/s{R}\n")
            return

        # show any text alongside tool calls
        render_answer(resp)

        # add assistant message
        messages.append({"role": "assistant", "content": resp})

        # execute each tool call
        for call in calls:
            name = call["name"]
            args = call.get("arguments", {})
            args_str = json.dumps(args)
            if len(args_str) > 70:
                args_str = args_str[:67] + "..."

            print(f"  {MAGENTA}{B}⚡{R} {B}{name}{R} {GRAY}{args_str}{R}")

            result = execute_tool(name, args)
            result = truncate(result)

            # show result
            print(f"  {D}{'─' * 56}{R}")
            for line in result.split("\n"):
                print(f"  {GRAY}│{R} {line}")
            print(f"  {D}{'─' * 56}{R}\n")

            # feed result back
            messages.append({
                "role": "user",
                "content": f"Tool result for `{name}`:\n{result}"
            })

    print(f"  {RED}⚠ Max turns ({MAX_TURNS}) reached{R}\n")


def build_system_prompt():
    tools_desc = ""
    for name, t in TOOLS.items():
        schema = json.dumps(t["schema"], indent=2)
        tools_desc += f"### {name}\n```json\n{schema}\n```\n\n"
    header = "You are GLM-Agent, an autonomous coding assistant. You have access to tools.\n\n"
    header += "## How to Use Tools\n\n"
    header += 'To call a tool, output a JSON block like this:\n\n'
    header += '```json\n{"name": "tool_name", "arguments": {"key": "value"}}\n```\n\n'
    header += "You can call multiple tools. Show reasoning before calling tools.\n\n"
    header += "## Available Tools\n\n"
    rules = "\n## Rules\n\n"
    rules += "1. Think step by step. Show your reasoning before calling tools.\n"
    rules += "2. Explore before making changes - read files first, glob to find things.\n"
    rules += "3. Test your changes with shell commands after making them.\n"
    rules += "4. When done, give a clear final answer (no tool call needed).\n"
    rules += "5. For long files, use offset/limit parameters.\n"
    rules += "6. Use web_search for recent info, docs, or unknowns.\n"
    rules += "7. Use memory tool to save important context that persists across sessions.\n"
    rules += "8. Be concise in your reasoning. Be thorough in your work.\n\n"
    rules += f"## Working Directory: {os.getcwd()}\n"
    return header + tools_desc + rules


def main():
    flags = sys.argv[1:]
    prompt_args = []
    temperature = 0.7

    i = 0
    while i < len(flags):
        if flags[i] == "--temp" and i + 1 < len(flags):
            temperature = float(flags[i + 1])
            i += 1
        elif flags[i] == "--help":
            print(f"""
  {B}{CYAN}GLM-Agent{R} — autonomous coding assistant

  {B}Usage:{R}
    glm --agent                     interactive agent
    glm --agent "task"              one-shot task
    glm --agent --temp 0.3 "task"   set temperature

  {B}Interactive:{R}
    quit, exit, q                   exit
    reset                           clear history
    /system <text>                  append to system prompt
    /tools                          list available tools
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
    print(f"  {B}{CYAN}GLM-Agent{R} {D}· {len(TOOLS)} tools{R}")
    print(f"  {GRAY}  {cwd}{R}")
    print(f"  {GRAY}  {B}/help{R}{GRAY} for commands · {B}Ctrl+C{R}{GRAY} exit{R}\n")

    while True:
        try:
            user = input(f"  {CYAN}{B}?{R}  ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {GRAY}Bye.{R}")
            break

        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            print(f"  {GRAY}Bye.{R}")
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
        if user == "/help":
            print(f"""
  {B}Commands:{R}
    {B}/system <text>{R}   append to system prompt
    {B}/tools{R}           list tools
    {B}reset{R}            clear history
    {B}quit{R}             exit

  {B}Example tasks:{R}
    "list all python files and count lines"
    "create a flask hello world app"
    "search for latest llama.cpp release"
    "read src/main.py and fix the bug"
    "remember: we use pytest for testing"
""")
            continue

        agent_loop(llm, user, messages, temperature)


if __name__ == "__main__":
    main()

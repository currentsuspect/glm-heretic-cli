import re
import json


STRICT_RESPONSE_SUFFIX = """

Strict response mode:
- Output only the final answer.
- No explanation.
- No preface.
- No quotes unless the user explicitly asks for quotes.
- No markdown unless explicitly requested.
- Do not describe what you are about to do.
""".strip()


STRICT_TOOL_SUFFIX = """

Strict tool mode:
- If a tool call is requested, output exactly one fenced JSON block and nothing else.
- Do not include explanation before or after the JSON block.
- Use only a valid tool name.
- Include all required arguments.
- Do not narrate tool intent.
""".strip()


def detect_prompt_mode(user_text):
    lowered = user_text.lower()
    if re.search(r"\breply with exactly\b", lowered):
        return "strict_response"
    if "output only" in lowered or "exactly one word" in lowered or "one short sentence" in lowered:
        return "strict_response"
    if "tool call" in lowered or "json tool call" in lowered:
        return "strict_tool"
    return None


def apply_prompt_mode(system_text, mode):
    if mode == "strict_response":
        return system_text + "\n\n" + STRICT_RESPONSE_SUFFIX
    if mode == "strict_tool":
        return system_text + "\n\n" + STRICT_TOOL_SUFFIX
    return system_text


def exact_output_from_prompt(user_text):
    patterns = [
        r"reply with exactly:\s*(.+)$",
        r"output only:\s*(.+)$",
        r"answer in one short sentence exactly:\s*(.+)$",
    ]
    stripped = user_text.strip()
    for pattern in patterns:
        match = re.search(pattern, stripped, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().strip('"').strip("'")
    return None


def tool_call_from_prompt(user_text):
    stripped = user_text.strip()

    shell_match = re.search(r"output a shell tool call that runs\s+(.+?)[\.\n]?$", stripped, re.IGNORECASE)
    if shell_match:
        command = shell_match.group(1).strip().strip("`")
        return {"name": "shell", "arguments": {"command": command}}

    path_match = re.search(r"output a (git_diff|read) tool call for\s+(.+?)[\.\n]?$", stripped, re.IGNORECASE)
    if path_match:
        tool = path_match.group(1)
        path = path_match.group(2).strip().strip("`")
        return {"name": tool, "arguments": {"path": path}}

    return None


def apply_output_contract(user_text, raw_text, mode):
    if mode == "strict_response":
        exact = exact_output_from_prompt(user_text)
        if exact is not None:
            return exact
        return raw_text.strip()

    if mode == "strict_tool":
        tool_call = tool_call_from_prompt(user_text)
        if tool_call is not None:
            return "```json\n" + json.dumps(tool_call) + "\n```"
        return raw_text.strip()

    return raw_text.strip()

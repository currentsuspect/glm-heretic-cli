import json
import os
import re
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

from .agent import parse_tool_calls
from .install import FILE, REPO
from .prompt_modes import apply_output_contract, apply_prompt_mode, exact_output_from_prompt
from .runtime import load_llm


EVALS_DIR = Path(__file__).resolve().parents[2] / "evals"
TASKS_PATH = EVALS_DIR / "tasks.json"
REPORTS_DIR = EVALS_DIR / "reports"


def load_tasks():
    with TASKS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_case(llm, case):
    started = time.time()
    system_text = apply_prompt_mode(case["system"], case.get("mode"))
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": case["prompt"]},
        ],
        max_tokens=case.get("max_tokens", 128),
        temperature=0.0,
        top_p=0.9,
        stop=["<|endoftext|>"],
    )
    elapsed = time.time() - started
    raw = output["choices"][0]["message"]["content"].strip()
    text = apply_output_contract(case["prompt"], raw, case.get("mode"))
    if case.get("mode") == "strict_response" and exact_output_from_prompt(case["prompt"]) is None:
        rewrite = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "Answer the user's question directly. Output only the final answer. No explanation. No bullets. No markdown. One short sentence.",
                },
                {
                    "role": "user",
                    "content": case["prompt"],
                },
            ],
            max_tokens=20,
            temperature=0.0,
            top_p=0.9,
            stop=["<|endoftext|>"],
        )
        text = rewrite["choices"][0]["message"]["content"].strip()
    passed, detail = score_case(case, text)
    return {
        "id": case["id"],
        "kind": case["kind"],
        "passed": passed,
        "detail": detail,
        "response": text,
        "elapsed_seconds": round(elapsed, 2),
        "completion_tokens": output["usage"]["completion_tokens"],
        "prompt_tokens": output["usage"]["prompt_tokens"],
    }


def score_case(case, text):
    kind = case["kind"]
    if kind == "exact":
        expected = case["expected"]
        passed = text == expected
        return passed, f"expected exact `{expected}`"
    if kind == "contains":
        expected = case["expected_contains"].lower()
        passed = expected in text.lower()
        return passed, f"expected substring `{expected}`"
    if kind == "regex":
        pattern = case["expected_regex"]
        passed = re.search(pattern, text, re.DOTALL) is not None
        return passed, f"expected regex `{pattern}`"
    if kind == "tool_call":
        calls = parse_tool_calls(text)
        if not calls:
            return False, "no valid tool call parsed"
        first = calls[0]
        if first["name"] != case["expected_name"]:
            return False, f"expected tool `{case['expected_name']}`, got `{first['name']}`"
        for key, value in case.get("expected_args", {}).items():
            if first["arguments"].get(key) != value:
                return False, f"expected arg `{key}` to equal `{value}`"
        return True, "valid tool call"
    return False, f"unknown case kind `{kind}`"


def summarize(results):
    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round((passed / total) * 100, 1) if total else 0.0,
    }


def save_report(report):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = REPORTS_DIR / f"{timestamp}.json"
    latest = REPORTS_DIR / "latest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with latest.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path, latest


def print_summary(report, report_path):
    summary = report["summary"]
    print("GLM eval scorecard")
    print(f"backend:    {report['backend']}")
    print(f"model_path:  {report['model_path']}")
    print(f"pass_rate:   {summary['passed']}/{summary['total']} ({summary['pass_rate']}%)")
    print(f"report:      {report_path}")
    print()
    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['id']}  {result['detail']}")
        print(f"  response: {result['response'][:160]}")


def main():
    tasks = load_tasks()
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            raise SystemExit("Usage: python -m glm_cli.evals [task_limit]")
    if limit is not None:
        tasks = tasks[:limit]

    model_path = hf_hub_download(repo_id=REPO, filename=FILE)
    llm, backend = load_llm(model_path=model_path, n_ctx=4096, verbose=False)
    results = [run_case(llm, case) for case in tasks]
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": backend,
        "model_path": model_path,
        "summary": summarize(results),
        "results": results,
    }
    report_path, latest_path = save_report(report)
    print_summary(report, report_path)
    print()
    print(f"latest: {latest_path}")


if __name__ == "__main__":
    main()

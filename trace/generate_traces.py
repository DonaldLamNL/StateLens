from __future__ import annotations

import argparse
import json
import sys
import codecs
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def register_rot13_alias() -> None:
    """Ensure `str.encode('rot13')` works by registering an alias."""

    def _search_function(name: str):
        if name.lower() == "rot13":
            return codecs.lookup("rot_13")
        return None

    codecs.register(_search_function)


class FunctionTracer:
    """Collects per-line local variable snapshots for a target function."""

    def __init__(self, code_text: str, target_code_obj) -> None:
        # Buffer code lines so we can show original source in traces.
        self.code_lines = code_text.splitlines()
        self.target_code_obj = target_code_obj
        self.records: List[Tuple[int, str, Dict[str, str]]] = []
        self.frame_state: Dict[Any, Dict[str, Optional[int]]] = {}

    def trace_dispatch(self, frame, event: str, arg):
        if event == "call" and frame.f_code is self.target_code_obj:
            # Start tracing only for the target function.
            self.frame_state[frame] = {"last_line": frame.f_code.co_firstlineno}
            return self._trace_frame
        return None

    def _trace_frame(self, frame, event: str, arg):
        state = self.frame_state.get(frame)
        if state is None:
            state = {"last_line": None}
            self.frame_state[frame] = state

        if event == "line":
            if state["last_line"] is not None:
                self._record_line(frame, state["last_line"])
            state["last_line"] = frame.f_lineno
        elif event == "return":
            if state["last_line"] is not None:
                self._record_line(frame, state["last_line"])
                state["last_line"] = None
            self.frame_state.pop(frame, None)
        return self._trace_frame

    def _record_line(self, frame, lineno: int) -> None:
        line_text = self._get_line_text(lineno)
        locals_snapshot = {
            name: self._safe_repr(value)
            for name, value in frame.f_locals.items()
            if self._should_include_var(name)
        }
        self.records.append((lineno, line_text, locals_snapshot))

    def _get_line_text(self, lineno: int) -> str:
        idx = lineno - 1
        if 0 <= idx < len(self.code_lines):
            return self.code_lines[idx]
        return ""

    @staticmethod
    def _safe_repr(value: Any) -> str:
        try:
            return repr(value)
        except Exception:
            return f"<unreprable {type(value).__name__}>"

    @staticmethod
    def _should_include_var(name: str) -> bool:
        return not name.startswith("__") and not name.startswith(".")

    def format_records(self) -> str:
        formatted_lines: List[str] = []
        for lineno, code_text, locals_dict in self.records:
            line_repr = f"Line {lineno}: `{code_text}`"
            if locals_dict:
                locals_repr = ", ".join(
                    f"{k}={v}" for k, v in sorted(locals_dict.items())
                )
                line_repr = f"{line_repr}; {locals_repr}"
            formatted_lines.append(line_repr)
        return "\n".join(formatted_lines)


def execute_with_trace(
    code_text: str, function_name: str, function_call_expr: str
) -> Tuple[str, str]:
    namespace: Dict[str, Any] = {}
    compiled = compile(code_text, "<sample>", "exec")
    exec(compiled, namespace)
    target_function = namespace[function_name]

    tracer = FunctionTracer(code_text, target_function.__code__)

    def run_call():
        return eval(function_call_expr, namespace, {})

    previous_trace = sys.gettrace()
    sys.settrace(tracer.trace_dispatch)
    try:
        result = run_call()
    finally:
        sys.settrace(previous_trace)

    trace_text = tracer.format_records()
    return trace_text, repr(result)


def build_conversation(code_text: str, function_call: str, trace: str, result: str):
    user_prompt = (
        "\nGiven the code snippet:\n"
        "```python\n"
        f"{code_text}\n"
        "```\n"
        "and the function call with input arguments:\n"
        "```python\n"
        f"{function_call}\n"
        "```\n"
        f"Predict the exact output value for `{function_call}`, "
        "execute the program step by step before arriving at an answer within "
        "the tokens <think> and </think>, and output your prediction using the "
        f"special tokens <answer> {function_call} == ?? [/answer]. "
        "Ensure the provided expression syntax is correct!\n"
    )

    assistant_reply = (
        "<think>\n"
        f"{trace}\n"
        "</think>\n"
        "<answer>\n"
        f"{function_call} == {result}\n"
        "</answer>"
    )

    return {
        "system": "You are an expert Python programmer",
        "conversations": [
            {"from": "user", "value": user_prompt},
            {"from": "assistant", "value": assistant_reply},
        ],
        "mask": "user",
        "type": "VALUE_TO_TEXT",
    }


def process_samples(input_path: Path, output_path: Path, limit: Optional[int]) -> None:
    register_rot13_alias()

    written = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            if limit is not None and written >= limit:
                break
            try:
                item = json.loads(line)
                trace_text, result_repr = execute_with_trace(
                    item["code"], item["function_name"], item["function_call"]
                )
                conversation = build_conversation(
                    item["code"], item["function_call"], trace_text, result_repr
                )
                dst.write(json.dumps(conversation, ensure_ascii=False))
                dst.write("\n")
                written += 1
            except Exception as exc:
                print(
                    f"[WARN] Failed to process line {line_number}: {exc}",
                    file=sys.stderr,
                )
    print(f"Generated {written} conversation records at {output_path}")


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Generate traced conversations from cruxeval data."
    )
    parser.add_argument(
        "--input",
        default="cruxeval.jsonl",
        help="Path to the source JSONL file",
    )
    parser.add_argument(
        "--output",
        default="training_traces.jsonl",
        help="Destination JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of items to process",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    process_samples(input_path, output_path, args.limit)


if __name__ == "__main__":
    main()


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
        line_text, indent = self._get_line_text(lineno)
        locals_snapshot = {
            name: self._safe_repr(value)
            for name, value in frame.f_locals.items()
            if self._should_include_var(name)
        }
        self.records.append((lineno, line_text, locals_snapshot, indent))

    def _get_line_text(self, lineno: int) -> Tuple[str, int]:
        idx = lineno - 1
        if 0 <= idx < len(self.code_lines):
            raw = self.code_lines[idx]
            stripped = raw.lstrip()
            indent = len(raw) - len(stripped)
            # Remove indentation so Line records remain compact.
            return stripped, indent
        return "", 0

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

        class LoopState:
            __slots__ = (
                "key",
                "indent",
                "iterations",
                "current",
                "pending_header",
            )

            def __init__(self, key: Tuple[int, str], indent: int):
                self.key = key
                self.indent = indent
                self.iterations: List[List[str]] = []
                self.current: List[str] = []
                self.pending_header: Optional[str] = None

        loop_stack: List[LoopState] = []

        def render_line(lineno: int, code_text: str, locals_dict: Dict[str, str]) -> str:
            line_repr = f"Line {lineno}: `{code_text}`"
            if locals_dict:
                locals_repr = ", ".join(
                    f"{k}={v}" for k, v in sorted(locals_dict.items())
                )
                line_repr = f"{line_repr}; {locals_repr}"
            return line_repr

        def emit_lines(lines: List[str]):
            if not lines:
                return
            if loop_stack:
                loop_stack[-1].current.extend(lines)
            else:
                formatted_lines.extend(lines)

        def close_loop(state: LoopState, top_level: bool) -> List[str]:
            if state.current:
                state.iterations.append(state.current)
                state.current = []
            state.pending_header = None
            total = len(state.iterations)
            if total == 0:
                return []
            flattened: List[str] = []

            def append_iteration(iter_no: int, chunk: List[str]):
                if top_level:
                    flattened.append(f"=== Iteration {iter_no} ===")
                flattened.extend(chunk)

            def build_skip_line(skip_start: int, skip_end: int) -> str:
                if skip_end == skip_start:
                    core = f"Iterations {skip_start} skipped for `Line {state.key[0]}`"
                else:
                    core = f"Iterations {skip_start} to {skip_end} skipped for `Line {state.key[0]}`"
                if top_level:
                    return f"=== {core} ==="
                return f"... ({core})"

            if total <= 3:
                for idx, chunk in enumerate(state.iterations, start=1):
                    append_iteration(idx, chunk)
                return flattened

            append_iteration(1, state.iterations[0])
            append_iteration(2, state.iterations[1])
            skipped_end = total - 1
            flattened.append(build_skip_line(3, skipped_end))
            if top_level:
                flattened.append("...")
            append_iteration(total, state.iterations[-1])
            return flattened

        def flush_pending_headers(current_indent: int):
            for state in loop_stack:
                if (
                    state.pending_header is not None
                    and current_indent > state.indent
                ):
                    state.current.append(state.pending_header)
                    state.pending_header = None

        for lineno, code_text, locals_dict, indent in self.records:
            key = (lineno, code_text)
            rendered = render_line(lineno, code_text, locals_dict)

            flush_pending_headers(indent)

            while loop_stack and indent <= loop_stack[-1].indent and key != loop_stack[-1].key:
                finished = loop_stack.pop()
                emit_lines(close_loop(finished, top_level=len(loop_stack) == 0))

            is_loop_header = code_text.startswith(("for ", "while "))
            if is_loop_header:
                if loop_stack and loop_stack[-1].key == key:
                    current_state = loop_stack[-1]
                    if current_state.current:
                        current_state.iterations.append(current_state.current)
                    current_state.current = []
                else:
                    current_state = LoopState(key, indent)
                    loop_stack.append(current_state)
                current_state.pending_header = rendered
                continue

            flush_pending_headers(indent)

            if loop_stack:
                loop_stack[-1].current.append(rendered)
            else:
                formatted_lines.append(rendered)

        while loop_stack:
            finished = loop_stack.pop()
            emit_lines(close_loop(finished, top_level=len(loop_stack) == 0))

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
        f"special tokens <answer> {function_call} == ?? </answer>. "
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
        default="data_py_trace_skip.jsonl",
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


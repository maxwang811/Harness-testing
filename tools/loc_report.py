"""Goal-2 scorecard input: code size per implementation (non-blank, non-comment, non-docstring lines).

    python -m tools.loc_report
"""
from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "loop-ref (no framework)": ["impls/reference_loop.py"],
    "pydantic-ai slice": ["impls/pydantic_ai_slice.py"],
    "openai-agents slice": ["impls/openai_agents_slice.py"],
    "shared slice glue (both slices)": ["impls/common.py"],
}


def code_lines(path: Path, section: str | None = None) -> int:
    src = path.read_text()
    doc_lines: set[int] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
                doc_lines.update(range(first.lineno, first.end_lineno + 1))
    lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            if tok.start[0] not in doc_lines:
                lines.add(tok.start[0])
    if section:
        text = src.splitlines()
        start = next((i for i, l in enumerate(text, 1) if l.startswith("# ====") and section in l), None)
        end = next((i for i, l in enumerate(text, 1) if i > (start or 0) and l.startswith("# ====")), len(text) + 1)
        lines = {l for l in lines if start and start < l < end}
    return len(lines)


def main() -> None:
    for name, files in FILES.items():
        total = sum(code_lines(ROOT / f) for f in files)
        adapter = sum(code_lines(ROOT / f, "gateway adapter") for f in files if "slice" in f)
        extra = f"  (gateway adapter section: {adapter})" if adapter else ""
        print(f"{name:<34} {total:>4} lines{extra}")


if __name__ == "__main__":
    main()

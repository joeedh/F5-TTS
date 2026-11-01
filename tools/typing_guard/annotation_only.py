"""Check that files differ from HEAD only by annotations (docs/plans/python-pyright-tooling.md, step 4).

Usage, before committing a batch:  python tools/typing_guard/annotation_only.py <file.py> ...

Both versions are parsed and the following are removed before comparing the syntax trees:
parameter and return annotations, annotated assignments (kept as plain assignments),
imports, and the module-level aliases named in TYPE_ALIASES. Comments and formatting are
ignored. Exit code 1 if any file differs by anything else. This covers the files that the
numeric guard in golden.py cannot reach, such as the Gradio apps, trainer and eval code.
"""

from __future__ import annotations

import ast
import subprocess
import sys


# Module-level names that a batch may add as type aliases (for example Rope in model/modules.py).
TYPE_ALIASES = {"Rope", "_T"}


class StripAnnotations(ast.NodeTransformer):
    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if node.value is None:
            return None
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


def is_alias(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in TYPE_ALIASES
    )


def normalise(source: str) -> str:
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom)) and not is_alias(n)]
    return ast.dump(StripAnnotations().visit(tree))


def main(paths: list[str]) -> int:
    differing = 0
    for path in paths:
        result = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, check=True)
        with open(path, encoding="utf-8") as f:
            same = normalise(result.stdout.decode("utf-8")) == normalise(f.read())
        print(("annotation-only  " if same else "DIFFERS          ") + path)
        differing += not same
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

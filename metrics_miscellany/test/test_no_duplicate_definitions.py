import ast
import pathlib
from collections import Counter

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent


def _modules():
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def _top_level_definitions(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_duplicate_top_level_definitions(path):
    """No module may define the same top-level name twice.

    The literate source this package was tangled from through 0.3.x
    concatenated blocks sharing a :tangle target, which made this easy to
    do by accident: utils.leverage was defined twice, and the two
    definitions disagreed.  That source is gone, but ruff's F811 is still
    disabled package-wide (see [tool.ruff.lint.per-file-ignores]) to
    tolerate the repeated imports it left behind, so nothing else catches
    a shadowed definition.  Retire this test once those ignores go.
    """
    duplicates = {
        name: n for name, n in Counter(_top_level_definitions(path)).items() if n > 1
    }
    assert not duplicates, (
        f"{path.relative_to(PACKAGE.parent)} defines "
        + ", ".join(f"{name!r} {n} times" for name, n in sorted(duplicates.items()))
        + ".  The later definition wins and the earlier is dead code."
    )


if __name__ == "__main__":
    for p in _modules():
        test_no_duplicate_top_level_definitions(p)
    print("smoke OK")

"""Source conventions for the perception package."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCES = [p for p in sorted(PACKAGE_ROOT.rglob("*.py")) if p.parent.name != "test"]
CONTAINER_MODULES = {"collector.py", "labels.py", "scene_sampler.py", "schema.py", "config.py", "geometry.py", "layout.py"}
TORCH_MODULES = {"dataset.py", "model.py", "training.py", "estimator.py", "inference_node.py", "curation.py"}


def imported_modules(tree: ast.AST) -> set[str]:
    """Top-level module names imported by a parsed source file."""

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_every_module_and_callable_has_a_docstring() -> None:
    """Modules, classes, and functions are documented."""

    missing = []
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if ast.get_docstring(tree) is None:
            missing.append(f"{path.name}: module")
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and ast.get_docstring(node) is None:
                missing.append(f"{path.name}:{node.lineno} {node.name}")
    assert not missing, "\n".join(missing)


def test_no_leading_underscore_names() -> None:
    """Definitions use plain names; only dunder methods are allowed."""

    offenders = []
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                name = node.name
                if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
                    offenders.append(f"{path.name}:{node.lineno} {name}")
    assert not offenders, "\n".join(offenders)


def test_container_modules_avoid_torch_and_torch_modules_avoid_cv2() -> None:
    """Collection code runs in aic_eval without torch; pixi-side code avoids the cv2/torchvision libjpeg clash."""

    for path in SOURCES:
        modules = imported_modules(ast.parse(path.read_text(encoding="utf-8")))
        if path.name in CONTAINER_MODULES:
            assert "torch" not in modules, path
        if path.name in TORCH_MODULES:
            assert "cv2" not in modules, path

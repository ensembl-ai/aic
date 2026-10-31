"""Source-quality contract for the AIC MuJoCo Python code."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACKAGE = PACKAGE_ROOT / "aic_mujoco"
LEARNED_CONTROL_FILES = (
    RUNTIME_PACKAGE / "collection.py",
    RUNTIME_PACKAGE / "commands.py",
    RUNTIME_PACKAGE / "controllers.py",
    RUNTIME_PACKAGE / "dataset.py",
    RUNTIME_PACKAGE / "config.py",
    PACKAGE_ROOT / "configs" / "collect.json",
)


def test_all_source_modules_and_callables_have_docstrings() -> None:
    """Require documentation on every module, class, method, and function."""

    missing: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.parent.name == "test":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if ast.get_docstring(tree) is None:
            missing.append(f"{path}: module")
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    missing.append(f"{path}:{node.lineno}: {node.name}")
    assert not missing, "Missing docstrings:\n" + "\n".join(missing)


def test_active_simulation_architecture_rules_apply_package_wide() -> None:
    """Enforce script placement, naming, and orientation boundaries."""

    assert not (PACKAGE_ROOT / "collect_data.py").exists()
    assert (PACKAGE_ROOT / "scripts" / "collect_data.py").is_file()
    assert (RUNTIME_PACKAGE / "utils").is_dir()

    forbidden_quaternion_helpers = (
        "multiply_quaternion",
        "quaternion_multiply",
        "quaternion_product",
        "rotate_vector_wxyz",
    )
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.parent.name == "test":
            continue
        source = path.read_text(encoding="utf-8")
        for helper in forbidden_quaternion_helpers:
            assert helper not in source, f"Handwritten quaternion helper in {path}"

    forbidden_control_fields = (
        "goal_offset_quaternion",
        "goal_quaternion",
        "sfp_tip_quaternion",
        "quaternion_wxyz",
    )
    for path in LEARNED_CONTROL_FILES:
        source = path.read_text(encoding="utf-8")
        for field in forbidden_control_fields:
            assert field not in source, f"Quaternion control/dataset field in {path}"

    private_top_level: list[str] = []
    for path in sorted(RUNTIME_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    private_top_level.append(f"{path}:{node.lineno}: {node.name}")
    assert not private_top_level, "Unnecessary private top-level names:\n" + "\n".join(
        private_top_level
    )

"""Packaging contracts for a fresh repository-root uv invocation."""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _dependency_names(specifications: list[str]) -> set[str]:
    return {
        re.split(r"[<>=!~;\s\[]", specification, maxsplit=1)[0].lower()
        for specification in specifications
    }


def _production_imports() -> set[str]:
    imports: set[str] = set()
    for path in (PROJECT_ROOT / "src/hqrc_v3").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imports.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.partition(".")[0])
    return imports - sys.stdlib_module_names - {"hqrc_v3"}


def test_child_project_declares_an_exact_installable_src_package() -> None:
    project = _load_toml(PROJECT_ROOT / "pyproject.toml")

    assert project["build-system"]["build-backend"] == "setuptools.build_meta"
    assert project["build-system"]["requires"]
    assert project["project"]["requires-python"] == ">=3.11"
    assert project["project"]["scripts"] == {"hqrc": "hqrc_v3.cli:main"}
    assert project["tool"]["setuptools"]["packages"]["find"] == {
        "where": ["src"],
        "include": ["hqrc_v3", "hqrc_v3.*"],
        "namespaces": False,
    }


def test_every_production_import_has_a_declared_distribution() -> None:
    project = _load_toml(PROJECT_ROOT / "pyproject.toml")
    required = _dependency_names(project["project"]["dependencies"])
    optional = set().union(
        *(
            _dependency_names(specifications)
            for specifications in project["project"]["optional-dependencies"].values()
        )
    )
    import_distributions = {
        "sklearn": "scikit-learn",
        **{name: name for name in _production_imports() if name != "sklearn"},
    }

    assert set(import_distributions.values()) <= required | optional
    assert {
        "arviz",
        "lightgbm",
        "numpy",
        "pandas",
        "polars",
        "pymc",
        "pytensor",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "torch",
        "xarray",
        "xgboost",
    } <= required
    assert "nutpie" in optional
    assert {"pytest", "ruff"} <= _dependency_names(project["dependency-groups"]["dev"])


def test_root_workspace_lock_contains_the_installable_child() -> None:
    root_project = _load_toml(REPOSITORY_ROOT / "pyproject.toml")
    lock = _load_toml(REPOSITORY_ROOT / "uv.lock")

    assert root_project["tool"]["uv"]["workspace"]["members"] == ["hqrc_v3"]
    package = next(entry for entry in lock["package"] if entry["name"] == "hqrc-v3")
    assert package["source"] == {"editable": "hqrc_v3"}

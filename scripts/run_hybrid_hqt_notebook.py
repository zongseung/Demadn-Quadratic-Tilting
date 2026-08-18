"""Execute the canonical HP–Fourier–LSTM + quadratic HQT notebook."""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.validator import normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _code_cell_with(nb: nbformat.NotebookNode, needle: str) -> nbformat.NotebookNode:
    matches = [
        cell for cell in nb.cells if cell.cell_type == "code" and needle in cell.source
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one code cell containing {needle!r}; found {len(matches)}"
        )
    return matches[0]


def _replace_assignment(
    source: str, name: str, value: object, *, expression: bool = False
) -> str:
    replacement = f"{name} = {value if expression else repr(value)}"
    updated, count = re.subn(
        rf"^{re.escape(name)}\s*=.*$",
        replacement,
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError(f"could not patch notebook assignment {name}")
    return updated


def prepare_notebook(
    source_path: Path,
    output_dir: Path,
    *,
    chains: int,
    draws: int,
    tune: int,
    target_accept: float,
) -> nbformat.NotebookNode:
    """Patch only run-specific paths and budgets into the canonical notebook."""

    with source_path.open(encoding="utf-8") as handle:
        original = nbformat.read(handle, as_version=4)
    _, original = normalize(original)
    nb = copy.deepcopy(original)

    setup = _code_cell_with(nb, "OUTPUT_DIR =")
    setup.source = _replace_assignment(
        setup.source,
        "OUTPUT_DIR",
        f"pathlib.Path({str(output_dir)!r})",
        expression=True,
    )

    run_cell = _code_cell_with(nb, "HQT_CHAINS =")
    for name, value in (
        ("HQT_CHAINS", chains),
        ("HQT_DRAWS", draws),
        ("HQT_TUNE", tune),
        ("HQT_TARGET_ACCEPT", target_accept),
    ):
        run_cell.source = _replace_assignment(run_cell.source, name, value)
    return nb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=(
            PROJECT_ROOT / "ver2" / "hybrid_hqt" / "hqt_verification_source.ipynb"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "hybrid_hqt_notebook_quadratic",
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--tune", type=int, default=3000)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 2 chains with 100 tune and 100 posterior draws for a smoke run.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source_path = args.notebook.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chains, draws, tune, target_accept = (
        (2, 100, 100, 0.90)
        if args.quick
        else (args.chains, args.draws, args.tune, args.target_accept)
    )
    prepared = prepare_notebook(
        source_path,
        output_dir,
        chains=chains,
        draws=draws,
        tune=tune,
        target_accept=target_accept,
    )
    executed_path = output_dir / "hqt_verification_executed.ipynb"
    try:
        client = NotebookClient(
            prepared,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(PROJECT_ROOT)}},
            allow_errors=False,
            record_timing=True,
        )
        client.execute()
    finally:
        with executed_path.open("w", encoding="utf-8") as handle:
            nbformat.write(prepared, handle)
    print(executed_path)


if __name__ == "__main__":
    main()

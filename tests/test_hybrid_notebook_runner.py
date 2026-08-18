from pathlib import Path

import nbformat

from scripts.build_hybrid_hqt_notebook import build_notebook
from scripts.run_hybrid_hqt_notebook import prepare_notebook


ROOT = Path(__file__).resolve().parents[1]


def test_hybrid_notebook_preparation_is_reproducible(tmp_path: Path) -> None:
    notebook = prepare_notebook(
        ROOT / "ver2" / "hybrid_hqt" / "hqt_verification_source.ipynb",
        tmp_path,
        chains=2,
        draws=17,
        tune=19,
        target_accept=0.91,
    )
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    combined = "\n".join(sources)

    run_index = next(
        index for index, source in enumerate(sources) if "HQT_CHAINS" in source
    )

    assert "HQT_CHAINS = 2" in sources[run_index]
    assert "HQT_DRAWS = 17" in sources[run_index]
    assert "HQT_TUNE = 19" in sources[run_index]
    assert "HQT_TARGET_ACCEPT = 0.91" in sources[run_index]
    assert '"diagnostics": hqt.diagnostics' in combined
    assert str(tmp_path) in combined
    assert "type_curve_posterior_mw.csv" in sources[-1]
    assert "gamma" not in combined.lower()
    assert "γ" not in combined


def test_versioned_source_matches_notebook_builder() -> None:
    source_path = ROOT / "ver2" / "hybrid_hqt" / "hqt_verification_source.ipynb"
    with source_path.open(encoding="utf-8") as handle:
        versioned = nbformat.read(handle, as_version=4)
    generated = build_notebook()

    assert [cell.source for cell in versioned.cells] == [
        cell.source for cell in generated.cells
    ]

    for index, cell in enumerate(generated.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"<notebook-cell-{index}>", "exec")

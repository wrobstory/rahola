import importlib.util
from pathlib import Path

_BUILD_PDF = Path(__file__).parents[1] / "docs" / "paper" / "build_pdf.py"
_SPEC = importlib.util.spec_from_file_location("rahola_build_pdf", _BUILD_PDF)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_inline_preserves_both_candes_accents() -> None:
    source = "Patterson and Candès 2019; Gibbs and Candès 2021"
    assert _MODULE.inline(source) == r"Patterson and Cand\`es 2019; Gibbs and Cand\`es 2021"

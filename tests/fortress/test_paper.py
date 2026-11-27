from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"


def paper_sources() -> str:
    return "\n".join(path.read_text() for path in (PAPER / "sections").glob("*.tex"))


def test_paper_identity_and_frozen_evidence_scope() -> None:
    source = paper_sources()
    assert "Felix Devon" in source
    assert "felix.devon@shannon-ai.com" in source
    assert "SHANNON LAB LLC" in source
    assert "frozen" in source.lower()
    assert "7f31c2e0" in source
    assert "Full-grid collusion" in source
    assert "Second-party" in source or "second-party" in source
    assert "10.13140/RG.2.2.15046.31042" in source
    assert "10.13140/RG.2.2.15046.31042" in (ROOT / "CITATION.cff").read_text()
    assert "10.13140/RG.2.2.15046.31042" in (ROOT / "README.md").read_text()


def test_all_included_tex_sources_exist() -> None:
    main = (PAPER / "main.tex").read_text()
    for name in re.findall(r"\\input\{([^}]+)\}", main):
        path = PAPER / f"{name}.tex"
        assert path.is_file(), path


def test_all_citations_resolve() -> None:
    source = paper_sources()
    cited = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", source):
        cited.update(group.split(","))
    bibliography = (PAPER / "references.bib").read_text()
    defined = set(re.findall(r"@\w+\{([^,]+),", bibliography))
    assert cited <= defined, sorted(cited - defined)
    assert {"freivalds1977", "fips203", "fips204", "xurcroh-validation"} <= cited


def test_validation_cost_data_reconciles_to_public_total() -> None:
    with (PAPER / "data" / "validation-costs.csv").open() as handle:
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        total = sum(float(row["cost_usd"]) for row in rows)
    assert total == 25045.91


def test_all_vector_and_generated_figures_exist() -> None:
    required = {
        "architecture.tex",
        "share-grid.tex",
        "freivalds.tex",
        "compiler.tex",
        "memory.tex",
        "session.tex",
        "evidence-chain.tex",
        "multimodal.tex",
    }
    assert required <= {path.name for path in (PAPER / "figures").glob("*.tex")}
    assert len(list((PAPER / "figures" / "generated").glob("*.tex"))) == 5


def test_compiled_paper_and_clean_build_log() -> None:
    assert (PAPER / "xurcroh-fortress.pdf").is_file()
    log = (PAPER / "xurcroh-fortress.log").read_text()
    for pattern in ("undefined", "Overfull", "Missing character", "duplicate ignored"):
        assert pattern not in log

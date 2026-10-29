"""Operations code never sees the truth: it does not import the simulator or the evaluation and never
names the truth file or directory. Only src/cpa/sim, evaluation.py, and the orchestration that runs
them (pipeline, cli, heldout) may; settings.py defines the paths but reads nothing."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "cpa"
ALLOWED = {
    "sim",
    "evaluation.py",
    "export.py",
    "pipeline.py",
    "cli.py",
    "heldout.py",
    "settings.py",
}  # settings only names the paths
FORBIDDEN_IMPORTS = ("cpa.sim", "cpa.evaluation", ".sim", "..sim", ".evaluation", "..evaluation")
FORBIDDEN_TEXT = ("fleet.yaml", "TRUTH", "truth_config", "/truth")


def shipping_files():
    for f in SRC.rglob("*.py"):
        rel = f.relative_to(SRC).parts
        if rel[0] in ALLOWED:
            continue
        yield f


def test_shipping_code_does_not_import_the_simulator_or_evaluation():
    bad = []
    for f in shipping_files():
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = ("." * node.level) + (node.module or "")
                names = [mod] + [f"{mod}.{a.name}" for a in node.names]
                if any(
                    n.startswith(p) or n.endswith(".sim") or n.endswith(".evaluation")
                    for n in names
                    for p in FORBIDDEN_IMPORTS
                    if n.startswith(p)
                ):
                    bad.append(f"{f.relative_to(SRC)}: {mod}")
                if node.level and any(a.name in ("sim", "evaluation") for a in node.names):
                    bad.append(f"{f.relative_to(SRC)}: from {mod} import sim/evaluation")
            elif isinstance(node, ast.Import):
                bad += [
                    f"{f.relative_to(SRC)}: {a.name}"
                    for a in node.names
                    if a.name.startswith(FORBIDDEN_IMPORTS)
                ]
    assert not bad, bad


def test_shipping_code_does_not_name_the_truth():
    bad = [
        f"{f.relative_to(SRC)}: {t}" for f in shipping_files() for t in FORBIDDEN_TEXT if t in f.read_text()
    ]
    assert not bad, bad


def test_the_check_would_catch_a_leak(tmp_path):
    leak = "from ..sim import world\n"
    tree = ast.parse(leak)
    node = tree.body[0]
    assert node.level == 2 and node.module == "sim"

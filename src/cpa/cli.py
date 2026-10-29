"""cpa: build the world, train, run the policies, evaluate, gate; serve the live system."""

from __future__ import annotations

import json

import typer

from . import pipeline, settings

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def run() -> None:
    """Everything, ending with the release gate (exit 1 if it fails)."""
    out = pipeline.run()
    raise typer.Exit(0 if out["gate"]["passed"] else 1)


@app.command()
def world() -> None:
    """Build the fleet and simulate the history under today's process."""
    pipeline.world()


@app.command()
def train() -> None:
    """Train, check and register the first champion models."""
    pipeline.train()


@app.command()
def runs(only: list[str] = typer.Option(None, "--only"), workers: int = 2) -> None:  # noqa: B008
    """Run the policies over the test weeks."""
    pipeline.runs(only or None, workers=workers)


@app.command()
def evaluate() -> None:
    """Judge the runs against the truth and apply the release gate."""
    from .gate import gate

    res = pipeline.evaluate()
    v = gate(res)
    pipeline._json(settings.REPORTS / "gate.json", v)
    for c in v["checks"]:
        typer.echo(f"{'ok  ' if c['passed'] else 'FAIL'} {c['id']}: {c['detail']}")
    raise typer.Exit(0 if v["passed"] else 1)


@app.command()
def apim(out: str = "azure/apim/policies") -> None:
    """Export the gateway's use cases as Azure API Management AI-gateway policies."""
    from .gateway import apim as A

    for f in A.export(out):
        typer.echo(f)


@app.command()
def heldout(name: str) -> None:
    """Run a held-out set of worlds (first run recorded once, never rewritten)."""
    from . import heldout as H

    typer.echo(json.dumps(H.run_set(name)["summary"], indent=1))


if __name__ == "__main__":
    app()

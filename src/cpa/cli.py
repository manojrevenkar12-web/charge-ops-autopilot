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
def runs(only: list[str] = typer.Option(None, "--only"), workers: int = 0) -> None:  # noqa: B008
    """Run the policies over the test weeks."""
    pipeline.runs(only or None, workers=workers or None)


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


@app.command("export-web")
def export_web() -> None:
    """Write the operations room's data (web/public/data)."""
    from .export import export

    typer.echo(", ".join(export()))


@app.command("serve-api")
def serve_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    """The model and gateway API (cpa.service.app)."""
    import uvicorn

    from .mlops.registry import Registry
    from .service.app import create_app

    uvicorn.run(create_app(Registry()), host=host, port=port)


@app.command("serve-csms")
def serve_csms(host: str = "0.0.0.0", port: int = 9000) -> None:
    """The OCPP 2.0.1 central system (PostgreSQL record when CPA_DATABASE_URL is set)."""
    import asyncio
    import os
    import time

    from .csms.server import CSMS, Clock
    from .csms.store import MemoryStore, PostgresStore

    async def main() -> None:
        url = os.environ.get("CPA_DATABASE_URL")
        store = await PostgresStore(url).open() if url else MemoryStore()
        csms = CSMS(Clock(int(time.time())), store)
        p = await csms.serve(host, port)
        typer.echo(f"CSMS on ws://{host}:{p}/<station id> (ocpp2.0.1)")
        await asyncio.Future()

    asyncio.run(main())


@app.command()
def heldout(name: str) -> None:
    """Run a held-out set of worlds (first run recorded once, never rewritten)."""
    from . import heldout as H

    typer.echo(json.dumps(H.run_set(name)["summary"], indent=1))


if __name__ == "__main__":
    app()

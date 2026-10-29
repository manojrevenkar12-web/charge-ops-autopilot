# Working in this repository

## Fast feedback

- `uv sync`, then `uv run pytest -q -m "not slow"` (seconds) and `uv run ruff check src tests`.
- `uv run pytest -q` adds the slow tests (a two-day simulation for training/serving parity) and the
  PostgreSQL store test (`CPA_TEST_DATABASE_URL`).
- `uv run cpa run` builds the fleet and its history, trains and registers the first champions, runs
  every policy over the same test weeks, evaluates against the truth and ends with the release gate
  (`config/release.yaml`); about 40 minutes on four cores.
- `uv run cpa export-web`, then `cd web && npm test && npx playwright test` for the operations room.

## Invariants (each has a test or a gate check)

- Truth isolation: only `src/cpa/sim` and `src/cpa/evaluation.py` read `build/truth` or
  `config/fleet.yaml`. Operations code (`ops`, `ml`, `mlops`, `gateway`, `csms`, `llm`, `kb`) never
  imports the simulator or the evaluation (`tests/test_isolation.py`).
- Models read, code decides: the planner turns model outputs into an action; the shield has the last
  word on every action, whoever proposed it; the LLM's recommended action is never executed as it
  stands (`llm-decides` is the positive control that shows why, `harm-is-measurable` in the gate).
- Every model call goes through the gateway (`cpa.gateway.core.Gateway.call`): use cases, budgets,
  limits, PII masking, quarantine of instruction-like untrusted text, output schemas.
- Training and serving compute features with the same code (`tests/test_features.py`).
- A model decides only after the offline gate, the ONNX check and a shadow period
  (`config/models.yaml`); every version is in the MLflow registry with its model card.
- Thresholds, rules with sources, prices and budgets live in `config/`, not in code. The APIM policies
  in `azure/apim` are generated from `config/gateway.yaml` (CI fails if they are stale).
- The central system's OCPP record and the command log are append-only (`sql/csms.sql`).

## Held-out sets

`heldout/<set>/fleets/*.yaml` are written blind from `heldout/WRITING_SCENARIOS.md` by someone who has
not read the operations code. Never change the operations code to do better on a set. Its first run
(`reports/heldout/<set>/first_run.json`) is never rewritten. If you study a set's failures to change
the code, write `regression` into `heldout/<set>/STATUS` and have someone write a new set.

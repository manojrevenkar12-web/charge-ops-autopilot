# 4. A model version decides only after a gate, ONNX parity and a shadow period

Status: accepted

## Context
Data scientists' models arrive as training code; in production they must be versioned, reproducible,
fast, monitored and replaceable without a bad version ever deciding.

## Decision
Every version goes into the MLflow registry with its model card, metrics and checks. A challenger
must not be worse than the champion on the same most recent weeks (overall and per hardware model),
must give the same answers in ONNX Runtime (top-class agreement at least 99.9%, p99 difference below
1e-4) and fit the latency budget; then it is scored alongside the champion on incidents as they
resolve and is promoted only if it is right at least as often. Retraining runs weekly inside the
autopilot (and as an Azure ML pipeline in production); drift of the features the model sees is
checked daily. Training and serving compute features with the same code, and a test compares them
hour by hour.

## Consequences
New versions take a week or more to decide anything. That is the point.

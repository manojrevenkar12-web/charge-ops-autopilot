# 2. Models read, code decides, the shield has the last word

Status: accepted

## Context
Vendor agents (AMPECO CoOperator, Monta's NOC agent) diagnose and recommend; a person approves
every action. An in-house AI unit can go further, but an LLM that executes its own recommendation
can be steered by text from chargers (techInfo is vendor-controlled free text) and cannot be
audited rule by rule.

## Decision
- The diagnosis model and the LLM reading only change probabilities. The planner turns them into
  an action with an expected success taken from Beta beliefs that start from the manuals' wording
  and are updated by observed outcomes.
- The shield checks every action, whoever proposed it, against rules with sources (no reset on an
  insulation fault, no immediate reset during a session, cooldowns, one reset per site at a time,
  network rate limits, staged firmware, an allowlist).
- Autonomy is earned per action and hardware model: proposals at first, acting alone once the
  Wilson lower bound of the observed success rate passes 0.6 over at least 15 outcomes; back to
  proposals if it falls. Firmware is never autonomous.
- `llm-decides` executes the model's recommended action as it stands. It is a positive control:
  the release gate requires it to show harm, or the harm measurement itself is broken.

## Consequences
The autopilot is slower to act than an agent that trusts its model, and faster than one that asks
a person every time; the comparison is in the results.

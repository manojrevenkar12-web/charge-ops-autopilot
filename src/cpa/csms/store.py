"""Where the central system records every OCPP message: PostgreSQL (append-only) in production and in
the DB tests, or memory for quick runs. Both return the same flat rows."""

from __future__ import annotations

import json
from typing import Any, Protocol

import polars as pl

from .. import ocpp_flat
from ..settings import ROOT


class Store(Protocol):
    async def append(self, rows: list[dict[str, Any]], payload: dict[str, Any]) -> None: ...

    async def between(self, lo: int, hi: int) -> pl.DataFrame: ...

    async def command(self, rec: dict[str, Any]) -> None: ...


class MemoryStore:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.payloads: list[dict[str, Any]] = []
        self.commands: list[dict[str, Any]] = []

    async def append(self, rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
        self.rows.extend(rows)
        self.payloads.append(payload)

    async def between(self, lo: int, hi: int) -> pl.DataFrame:
        sel = [r for r in self.rows if lo <= r["received"] < hi]
        return pl.DataFrame(sel, schema=ocpp_flat.SCHEMA) if sel else ocpp_flat.empty()

    async def command(self, rec: dict[str, Any]) -> None:
        self.commands.append(rec)


class PostgresStore:
    def __init__(self, url: str) -> None:
        self.url = url
        self.conn: Any = None

    async def open(self) -> PostgresStore:
        import psycopg

        self.conn = await psycopg.AsyncConnection.connect(self.url, autocommit=True)
        await self.conn.execute((ROOT / "sql" / "csms.sql").read_text())
        return self

    async def append(self, rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
        async with self.conn.cursor() as cur:
            for r in rows:
                await cur.execute(
                    "insert into ocpp_message (ts, received, charger_id, direction, action, message_id, payload, flat)"
                    " values (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        r["ts"],
                        r["received"],
                        r["charger_id"],
                        r["direction"],
                        r["action"],
                        r["message_id"],
                        json.dumps(payload),
                        json.dumps(r),
                    ),
                )

    async def between(self, lo: int, hi: int) -> pl.DataFrame:
        async with self.conn.cursor() as cur:
            await cur.execute(
                "select flat from ocpp_message where received >= %s and received < %s order by id", (lo, hi)
            )
            rows = [r[0] for r in await cur.fetchall()]
        return pl.DataFrame(rows, schema=ocpp_flat.SCHEMA) if rows else ocpp_flat.empty()

    async def command(self, rec: dict[str, Any]) -> None:
        await self.conn.execute(
            "insert into command (ts, charger_id, action, evse, params, decided_by, incident, reason, shield, result)"
            " values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                rec["ts"],
                rec["charger_id"],
                rec["action"],
                rec["evse"],
                json.dumps(rec["params"]),
                rec["decided_by"],
                rec["incident"],
                rec["reason"],
                json.dumps(rec["shield"]),
                rec["result"],
            ),
        )

    async def close(self) -> None:
        await self.conn.close()

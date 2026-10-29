"""A keyword reader for technicians' notes in ten languages: the cheap baseline the LLM and the
distilled model are compared against, and the stand-in provider when no model key is set."""

from __future__ import annotations

import re

PART_WORDS: dict[str, list[str]] = {
    "lock_actuator": ["lock", "verriegel", "blocco", "vergrendel", "lås", "blokad", "zámek", "zámku", "retesz",
                      "blocaj", "locking"],
    "power_module": ["power module", "leistungsmodul", "modulo di potenza", "vermogensmodule", "effektmodul",
                     "moduł mocy", "výkonový modul", "teljesítménymodul", "modulul de putere"],
    "rfid_reader": ["rfid"],
    "payment_terminal": ["payment", "bezahlterminal", "pagamento", "betaalterminal", "betalterminal",
                         "betalingsterminal", "płatnicz", "platební", "fizetési", "plată"],
    "cable": ["cable", "kabel", "cavo", "kábel", "cablu", "kablet", "kabeln", "insulation", "isolation", "izolac"],
    "modem": ["modem"],
}
RESTART_WORDS = ["power-cycled", "neu gestartet", "riavviata", "herstart", "omstartad", "genstartet", "restart",
                 "újraindítva", "repornit"]
NONE_WORDS = ["no fault", "kein fehler", "nessun guasto", "geen storing", "inget fel", "ingen fejl",
              "nie stwierdzono", "žádná závada", "hiba nem", "nu s-a găsit", "grid", "netzausfall", "blackout",
              "netstoring", "nätavbrott", "netudfald", "awaria sieci", "výpadek", "hálózati", "pană de rețea"]


def read(note: str, lang: str = "") -> dict[str, str | None]:
    text = note.lower()
    text = re.split(r"(part needed|ersatzteil nötig|ricambio necessario|onderdeel nodig|reservdel behövs|"
                    r"reservedel nødvendig|potrzebna część|potřebný díl|alkatrész szükséges|piesă necesară)", text)[0]
    for part, words in PART_WORDS.items():
        if any(w in text for w in words):
            return {"part": part, "cause": part}
    if any(w in text for w in NONE_WORDS):
        return {"part": None, "cause": "none"}
    if any(w in text for w in RESTART_WORDS):
        return {"part": None, "cause": "restart"}
    return {"part": None, "cause": "unknown"}

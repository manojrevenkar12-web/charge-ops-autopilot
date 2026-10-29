# Nordvolt DC-150 technical reference (excerpt)

## Fault codes (techCode)
- 0x2A07: locking pin position error. Remote unlock usually releases it. Repeated errors mean the lock actuator is worn (part: lock actuator).
- 0x3110: power module thermal derating. Not recoverable remotely; replace the power module.
- 0x1F01: ground fault / isolation fault. Safety stop. Never reset remotely; site visit with a new cable.
- PT-OFFLINE: payment terminal not reachable. Station reset; otherwise replace the payment terminal.

## Notes
- Firmware 2.12.1 improves OCPP reconnection. Cards not being read while app starts work points to the RFID reader (part: RFID reader); a station reset helps in about half the cases.

# Alpenstrom HPC-350 service manual (excerpt)

## Error codes
- E-LCK-31: connector lock feedback timeout. Send UnlockConnector for the EVSE; if the lock does not release, replace the lock actuator (part: lock actuator).
- E-PWR-12: power stage over-temperature, output limited. A reset clears the latch for a few hours at most; the power module must be replaced (part: power module). Rising module temperature over two to five days precedes this code.
- E-ISO-02: insulation monitoring device (IMD) trip. Do not reset remotely and do not re-energise. The cable or inlet is damaged; dispatch with a replacement cable.
- PT-OFFLINE: payment terminal lost its acquirer link. Reset the station; if it persists, replace the terminal.

## Firmware
- 3.4.2 is the long-term release. 3.5.0 adds ISO 15118-20 support; report any rise in EV communication errors after installing it to Alpenstrom support.

## Controller
- The main controller can stall while the station reports Available ("silent stall"); watchdog warnings in NotifyEvent precede it. Reset the EVSE, then the station.

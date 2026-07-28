# DGX Spark PD-controller wedge

Platform-level low-power latch on GB10 hardware (DGX Spark, ASUS Ascent
GX10, Dell partner GB10). Independent from the NCCL hang investigation
in `nccl-timeout-investigation.md`.

## Symptom

The whole SoC — Grace CPU **and** Blackwell GPU, sharing one power
budget — gets clamped to a low-power envelope. Recognized by:

- GPU graphics clock pinned at ~600–650 MHz instead of the Application
  Clock ceiling of 2418 MHz.
- GPU power draw stuck at ~10–18 W under load.
- Performance State reports `P0`.
- **Every** throttle-reason flag reports `Not Active` (no SW Power Cap,
  no thermal, no HW Slowdown).
- CPU-bound workloads (e.g. tokenization, dataloader prep) also run
  several times slower than on a healthy peer node. The same PD MCU
  clamps the Grace CPU's power, not just the GPU.

Confirmed once on this fleet: training step throughput on the wedged
host dropped to ~26% of normal, and tiktoken `encode_batch` on the
wedged host took 9+ min for what the healthy peer did in <3 min.

## Cause

The USB-C Power Delivery (PD) MCU lives inside the power brick, not on
the GB10 board. It can latch into a fault state that survives soft
reboots — power keeps flowing during a reboot, so the MCU's volatile
state is not cleared. NVIDIA's driver/firmware accounting layer doesn't
see this throttle, which is why every `nvidia-smi` throttle flag stays
green.

## Fix

Two-step procedure, both required for a durable fix:

1. **Cold-drain the brick.** Shut the box down, unplug the brick from
   the wall, wait 2–3 minutes (the 30-second minimum cited by some
   reports is not always sufficient), reconnect, boot. This clears the
   PD MCU's latched state.
2. **Flash PD firmware to 0x507 or newer**, then perform a *second*
   cold drain so the new firmware activates from EEPROM. Use
   `sudo fwupdmgr get-devices` / `get-upgrades` / `update`. Each vendor
   has its own track (ASUS / NVIDIA-reference / Dell partner do not
   share capsules); use the one offered by `fwupdmgr` on your box.

ESRT version readback after the flash can be wrong even when the flash
succeeded — verify by checking that GPU clocks hold at ~2418 MHz under
sustained load rather than by re-reading the version field.

## Post-driver-update cold-cycle protocol

**Run a full cold cycle after every NVIDIA driver update**, not just a
reboot. Driver 580.159.03 specifically has been reported by multiple
users as a trigger for the wedge (and is the version this fleet is on).
The pattern: driver update lands → soft reboot leaves EC/PD policy
state inconsistent with the new driver → first sustained workload
latches the PD MCU into the low-power state.

Practical: prefer `sudo apt dist-upgrade` over plain `apt upgrade`
(the latter can leave nvidia-* packages "kept back"). After any update
that touches `nvidia-driver*`, `nvidia-modprobe`, `nvidia-settings`, or
the CUDA toolkit, schedule a cold drain at the next maintenance
window. Don't kick off long training runs immediately after a driver
update — verify clocks first.

## Pre-flight check

Wedge state isn't visible to the OS until you run a sustained workload.
Detect it before starting a training run: run a brief GPU warmup, then
read `clocks.current.graphics`. A healthy box pins near 2418 MHz under
load; a wedged box stays near 637 MHz. Threshold around 1800 MHz
catches the wedge reliably without false positives during ramp-up.

Existing tools that do this:

- [hoesing/spark-gpu-throttle-check](https://github.com/hoesing/spark-gpu-throttle-check)
- [parallelArchitect/spark-gpu-throttle-check](https://github.com/parallelArchitect/spark-gpu-throttle-check)
  (enhanced fork, NVML-direct, JSON output)

Wire either into `ray_train.py` before `trainer.fit()` and fail the
run if any worker is wedged.

## Other known triggers

- **GUI/display manager disabled** (`systemctl disable gdm` +
  `set-default multi-user.target`). Independent from the driver-update
  trigger; the GB10 display stack ties into GPU power-state management
  even on headless installs. Leave GDM running and don't use the
  display. Alternative: unplug the HDMI cable entirely (changes the
  EDID handshake path and avoids the wedge). Re-enabling GDM is the
  better-understood fix.

## References

- [Sggin1 — GX10 PD Throttle Fix](https://github.com/Sggin1/DGX-SPARK/blob/main/GX10_PD_Throttle_Fix.md)
  — verified end-to-end procedure (cold drain → flash 0x507 → second
  cold drain), vendor firmware-track table.
- [dredyson — My Another Asus GX10 Problem Journey](https://dredyson.com/my-another-asus-gx10-problem-journey-what-i-learned-after-6-months-the-complete-proven-fix-guide-for-the-nvidia-gb10-intermittent-gpu-power-cap-pd-throttle-wedge-and-low-clock-trap-that-n/)
  — 6-month investigation, ranked fix list, GDM/HDMI workarounds,
  pre-flight tooling overview.
- [NVIDIA forum — DGX Spark GB10 GPU Trapped in 15W / 650MHz](https://forums.developer.nvidia.com/t/dgx-spark-grace-blackwell-gb10-performance-drop-gpu-trapped-in-15w-650mhz-loop-with-50-c-artificial-t-limit-temp/370304)
  — canonical symptom report with full `nvidia-smi` evidence (May 2026).
- [NVIDIA forum — Another Asus GX10 Problem](https://forums.developer.nvidia.com/t/another-asus-gx10-problem/371201)
  — driver 580.159.03 correlation (Lutti, post #3), preflight watchdog
  pattern (sggin1, post #4).
- [NVIDIA forum — Investigating 513MHz cap](https://forums.developer.nvidia.com/t/investigating-513mhz-cap-for-gpu/361296)
  — earlier (Feb 2026) report of the same wedge; same `Not Active`
  throttle-flag signature.

## What this does NOT explain

This is a compute-throughput throttle, distinct from the long-running
NCCL `ALLREDUCE` watchdog timeout investigated in
`nccl-timeout-investigation.md`. The proxy-pool race documented there
manifests as a *collective operation hang*, not a slow-but-progressing
workload. Both can occur on the same fleet and are not the same bug.
However, runs conducted while a PD wedge was live may have had timing
characteristics (per-step duration, NCCL queue depths) different from
runs on healthy hardware, so prior NCCL-hang timing measurements should
be treated with some uncertainty.

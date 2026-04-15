# NCCL ALLREDUCE Timeout / SIGSEGV on Dual DGX Spark

Ongoing investigation into long-running training hangs on a 2-node DGX Spark
(GB10 / Blackwell / aarch64) cluster connected via direct QSFP56 RoCE.

## 1. Problem statement

When running distributed training of the vanilla transformer over 2 DGX Spark
nodes (world size 2, one GPU per node), a single collective eventually stalls
after hours-to-days of successful training. The symptom is always the same
shape:

- One rank's watchdog fires on a stuck `ALLREDUCE` with
  `ran for 1800002 milliseconds before timing out` (the 30-minute
  `ProcessGroupNCCL` timeout).
- The other rank receives the flight-recorder dump signal over TCPStore,
  begins its dump, and a minute or so later dies with `SIGSEGV` inside
  `ncclLocalOpAppend()` reached via the DDP autograd hook path.
- Ray Train then tears the worker group down with
  `SYSTEM_ERROR ... connection error code 2. End of file.`

Runs succeed from a few hundred thousand to nearly two million collectives
before failing, so the failure is not tied to a specific iteration count or
elapsed time.

## 2. Environment

| | |
|---|---|
| Nodes | 2 × Gigabyte DGX Spark (GB10 SoC, Blackwell GPU, aarch64) |
| Host IPs (recent runs) | `aitopatom-0512` 192.168.200.13, `aitopatom-09a5` 192.168.200.12 |
| Host IPs (earlier runs) | 192.168.200.10 / 192.168.200.11 (same hardware, different addressing) |
| Interconnect | Direct QSFP56 / 200 GbE RoCE, ConnectX-7 dual-port (`rocep1s0f1`, `roceP2p1s0f1`) |
| Management LAN | `enP7s7` onboard RJ45, `10.9.9.0/24` |
| CUDA driver | 13000 (13.0) |
| NCCL | 2.28.9+cuda13.0 |
| PyTorch | bundled with the project's uv venv (torch lib path under
  `/tmp/ray/session_*/runtime_resources/.venv/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so`) |
| Framework | Ray Train `TorchTrainer` + `TorchConfig(backend="nccl")`, DDP, bf16 |
| Model | Encoder-decoder transformer, WMT EN-FR, bucket sizes produce ~15-20 gradient buckets per backward |
| GDR | Disabled on DGX Spark per NVIDIA FAQ — expected, NCCL logs `GPU Direct RDMA Disabled for HCA 0/1` |
| `ulimit -s` | soft 8 MiB, hard unlimited on both nodes (`/proc/<pid>/limits` verified) |

## 3. Crash signature (canonical)

Two things to explain:

1. **The stall.** Both ranks go silent *inside* the same backward pass for 30 minutes.
2. **The SIGSEGV.** ~60 s after the peer's watchdog fires and aborts its
   comm, the stalling rank dies in `ncclLocalOpAppend`.

### 3.1 The stall, from the diagnostic tails (run 4)

`get_train_debug_logger` in `trainable_architecture.py` writes a line at
every phase boundary of `train_epoch`. The run-4 tails localise the
stall to within one step:

```
# rank 1 (SIGSEGVs)
03:56:46,875 epoch=2 step=48161 batch=75575 before_backward
# (nothing else — never reaches after_backward)

# rank 0 (watchdog fires)
03:56:47,228 epoch=2 step=48161 batch=75575 after_backward
04:27:47,288 epoch=2 step=48161 batch=75575 before_optimizer   ← 31-minute gap
04:27:47,293                                    after_optimizer
```

So:

- **Rank 1 is stuck inside `loss.backward()`.** Inside backward, DDP's
  `Reducer::autograd_hook` fires `pncclAllReduce` on each bucket as it
  becomes ready. The SIGSEGV stack (§3.2) shows rank 1's autograd thread
  eventually in `ncclLocalOpAppend`, so at minimum the stall resolves
  *there*; the most parsimonious reading is that rank 1 was in
  `ncclLocalOpAppend` for the entire 30 minutes (the slow-path spin
  loop at `proxy.cc:496` can spin forever under `sched_yield()`).
- **Rank 0 is stuck in `accumulated_loss.item()`** — the only CUDA work
  between its `after_backward` and `before_optimizer` log lines is
  `accumulated_loss += loss.detach()` followed by `.item()`
  (`trainable_architecture.py:160,163`). `.item()` does a
  `cudaStreamSynchronize` on the compute stream, which transitively
  waits on the NCCL stream via `cudaStreamWaitEvent`s inserted by
  DDP's `Reducer::finalize_backward → Future::wait →
  synchronizeWithCurrentStreams` during the preceding backward. Rank
  0 is **not** stalled on its own — its GPU kernel for 1155874
  launched but cannot complete because rank 1 has not yet joined the
  allreduce. See §6.0 for the full stream-coupling chain and why GDR
  being off does not affect this.

Both stalls resume together at ~04:27:47, roughly when the peer watchdog
(04:26:47) fires and the resulting comm abort propagates.

### 3.2 The rank-1 dump response is ambiguous, not decisive

Rank 1's dump-response line (driver log line 242052) is:

```
[rank1] Received a dump signal due to a collective timeout from rank 0 ...
        Last enqueued NCCL work: 1155873, last completed NCCL work: 1155873
```

Earlier drafts of this doc read this as "rank 1 never called
`pncclAllReduce` for 1155874, so the stall must be outside NCCL."
**That inference is wrong.** In PyTorch v2.8 `ProcessGroupNCCL.cpp`,
`collective()` executes `C10D_NCCL_CHECK(fn(...))` (the call to
`pncclAllReduce`) at line 3695, and only *after* that call returns does
it call `workEnqueue(work)` at line 3751, which bumps
`pgStatus_->lastEnqueuedSeq` at line 3422. If the main thread is stuck
anywhere inside `pncclAllReduce` — including the full
`ncclEnqueueCheck → groupLaunch → hostStreamPlanTask → uploadProxyOps →
SaveProxy → ncclLocalOpAppend` chain — `lastEnqueuedSeq_` still shows
the previous successful collective. Combined with the rank-1
diagnostic tail ending at `before_backward`, the consistent reading is
that rank 1 **is** inside `pncclAllReduce` for collective 1155874, not
outside it.

The `last_enqueued = 1155888 / last_completed = 1155873` delta of 15 on
rank 0 is exactly one DDP backward pass' worth of gradient buckets —
rank 0 *did* complete its own backward (we have `after_backward`), so
all 15 allreduces were dispatched and `workEnqueue`d before the main
thread reached `.item()`. That's consistent with the normal DDP
pipeline depth, not with any NCCL leak or corruption.

### 3.3 The SIGSEGV

All runs with flight recorder enabled show the same head of the failing
stack on the rank that later SIGSEGVs:

```
PC: ncclLocalOpAppend()
    SaveProxy()
    ncclProxySaveOp()
    uploadProxyOps()
    hostStreamPlanTask()
    ncclLaunchKernelAfter_NoCuda()
    groupLaunch()
    ncclGroupEndInternal()
    ncclEnqueueCheck()
    pncclAllReduce
    c10d::ProcessGroupNCCL::collective<>()
    c10d::ProcessGroupNCCL::allreduce_impl()
    c10d::ProcessGroupNCCL::allreduce()
    c10d::ops::(anonymous)::allreduce_CUDA()
    ...
    c10d::Reducer::all_reduce_bucket()
    c10d::Reducer::mark_bucket_ready()
    c10d::Reducer::mark_variable_ready()
    c10d::Reducer::autograd_hook()
    torch::autograd::Engine::evaluate_function()
    torch::autograd::Engine::thread_main()
    torch::autograd::python::PythonEngine::thread_init()
```

The crashing thread is the autograd engine dispatching the next
allreduce from a bucket-ready hook. `ncclLaunchKernelAfter_NoCuda` is
called inline on this thread when `plan->isHostCbEnq` is false
(NCCL 2.28 `enqueue.cc:1649`), so this stack is *not* a CUDA host
callback and not the NCCL proxy progress thread — it is the autograd
worker itself, synchronously inside the NCCL enqueue path.

`ncclLocalOpAppend` in NCCL 2.28 (`proxy.cc:481-542`) takes a slot from
the per-rank free list. The fast path reads `proxyOps->freeOp` and
`op->next`; the slow path spins in
`while ((freeOp = pool->freeOps[tpLocalRank]) == -1) sched_yield();`
waiting for the proxy thread to refill. The spin cannot produce a
SIGSEGV by itself — a crash on this stack must be a pointer fault on one
of: `comm->proxyState->proxyOps`, `pool->ops + opIndex`, `op->next`, or
`pool->ops[…].next` when the proxy op pool has been freed or partly
re-initialised under us.

Two sub-cases are both consistent with this stack:

1. **The stall is in the slow-path spin loop itself.** The autograd
   thread has been spinning in `while ((freeOp = pool->freeOps[…]) ==
   -1) sched_yield();` for 30 minutes because the proxy progress thread
   has stopped refilling the free list. When the peer aborts and the
   proxy thread finally errors out of whatever network op was blocking
   it, it refills the list; the spin exits; the next instruction reads
   one of the pool fields (`pool->ops+opIndex`, `op->next`, or the
   head pointer `pool->ops[…].next`) while the pool itself is being
   torn down on another thread, and faults.
2. **The stall is downstream of the free-list acquire**, e.g. the
   autograd thread already holds a free slot and is blocked on a later
   step (`SaveProxy`, `memcpy`, or the CAS on `pool->nextOps`). Less
   likely given the shape of the function, but not excluded by the
   crash stack alone.

Both sub-cases share the same root cause: a stuck proxy op that the
proxy progress thread cannot drain. They differ only in which exact
instruction the autograd thread was parked on when the fault
happened, and either one agrees with the run-4 timeline:

| t (run 4)             | event |
|-----------------------|-------|
| 03:56:46.875          | rank 1 enters backward for batch 75575 (sync=True) |
| 03:56:47.105-228      | rank 0 runs backward for batch 75575 and returns |
| 03:56:47.228 → 04:27:47 | rank 0 stuck in `.item()`, rank 1 stuck inside backward / `ncclLocalOpAppend` |
| 04:26:47.344          | rank 0 watchdog fires, begins dump + abort |
| 04:26:47.572          | rank 1 TCPStore-handler reports `enq == comp == 1155873` |
| 04:27:47.288          | rank 0 `.item()` returns (stream unblocks on abort), proceeds to optimizer.step and several more no-sync batches |
| 04:27:54.898 (+67 s)  | rank 1 SEGVs in `ncclLocalOpAppend` |

**Signal chain:** a proxy op on some channel stops making progress →
both sides' proxy threads stall → rank 1's autograd thread blocks in
`ncclLocalOpAppend` slow path, rank 0's compute stream blocks in
`.item()` via its dependency on the NCCL stream → rank 0's 30-min
watchdog fires and aborts the comm → rank 0 unblocks and limps on,
rank 1 unblocks long enough to hit a freed proxy pool and SIGSEGV.

## 4. Per-run analysis

Five runs on disk in `logs/` (first four failed, fifth still running under
the current mitigation set).

### run_0 — baseline, no NCCL debug, no mitigation

| | |
|---|---|
| Log | `logs/run_0_driver_logs.txt` (8.5 MB) |
| Rank 0 host | 192.168.200.10 |
| Rank 1 host | 192.168.200.11 |
| Runtime before crash | ~52 h |
| Stalled collective (`SeqNum`) | ~1.72 M (largest observed to date) |
| Watchdog fires on | rank 1 (192.168.200.11) |
| SIGSEGV lands on | rank 0 (192.168.200.10) |
| Flight recorder | disabled → "Stack trace of the failed collective not found" |
| Debug env vars | none |

### run_1 — flight recorder enabled (legacy var)

| | |
|---|---|
| Log | `logs/run_1_driver_logs.txt` (3.5 MB) |
| Rank 0 host | 192.168.200.11 |
| Rank 1 host | 192.168.200.10 |
| Runtime before crash | ~20 h |
| Stalled collective | ~629 K |
| Watchdog fires on | rank 1 (192.168.200.10) |
| SIGSEGV lands on | rank 0 (192.168.200.11) |
| Full C++ backtrace captured | **yes** — canonical signature of §3 |
| Debug env vars | `TORCH_NCCL_TRACE_BUFFER_SIZE=1000` (legacy; deprecated in 2.28) |

First run that produced the full `autograd_hook → ncclLocalOpAppend` stack.

### run_2 — same config as run_1

| | |
|---|---|
| Log | `logs/run_2_driver_logs.txt` (5.3 MB) |
| Rank 0 host | 192.168.200.11 |
| Rank 1 host | 192.168.200.10 |
| Stalled collective | ~917 K (`SeqNum=917064`, `NumelIn=8395776`) |
| Watchdog fires on | rank 1 (192.168.200.10) |
| SIGSEGV lands on | rank 0 (192.168.200.11) |
| `last_enqueued - last_completed` on watchdog rank | 23 |
| Crash date | 2026-04-04 02:25 |

### run_4 — full NCCL debug, both RoCE ports active

| | |
|---|---|
| Log | `logs/run_4_driver_logs.txt` (37 MB) |
| Rank 0 host | 192.168.200.13 (`aitopatom-0512`), pid 327294 |
| Rank 1 host | 192.168.200.12 (`aitopatom-09a5`), pid 1600067 |
| Started | 2026-04-07 14:34 |
| Crashed | 2026-04-09 04:26 |
| Runtime | ~38 h |
| Stalled collective | `SeqNum=1155874`, `NumelIn=9445376`, `OpType=ALLREDUCE` |
| Watchdog fires on | rank 0 (192.168.200.13) |
| Rank 0 state at watchdog | last enqueued 1,155,888 / last completed 1,155,873 (gap = 15, one backward's worth of buckets) |
| Rank 1 state at dump | last enqueued = last completed = 1,155,873 — **consistent with being stuck inside `pncclAllReduce` for 1155874**, since `workEnqueue` bumps `lastEnqueuedSeq_` only *after* the NCCL call returns (see §3.2) |
| SIGSEGV lands on | rank 1 (192.168.200.12) |
| Timeline | 03:56:47 rank 0 issues 1155874 (30-min timer starts); 04:26:47.344 rank 0 watchdog fires; 04:26:47.572 rank 1 reports `enq==comp==1155873` in dump response; 04:27:54.898 rank 1 SIGSEGV in `ncclLocalOpAppend` (+67 s from watchdog) |
| Per-epoch batch counts | rank 0 local=251,184 / rank 1 local=265,072 → synced (min) 251,184. Rank 1's dataloader has ~5.5% more batches per epoch |
| NET | `Using [0]rocep1s0f1:1/RoCE [1]roceP2p1s0f1:1/RoCE [RO]; OOB enP7s7:10.9.9.250<0>` |
| Channels | 16 coll / 16 p2p |
| Crash epoch position | ~5 h 22 m into epoch 2 (epoch entered at 23:05:00, crash at 04:26:47). Not at an epoch boundary. |

This is the richest run. Both NICs and both `NCCL_DEBUG=INFO` + flight
recorder were active, producing the full `ncclCommInitRankConfig` trace,
a complete C++ backtrace at SIGSEGV, the rank-1 dump response showing
`last_enqueued == last_completed`, **and** the per-phase diagnostic tails
on both ranks that localise the stall to within one backward pass. See
§3.1 and §3.2.

The rank-1 > rank-0 batch-count skew is consistent across all three
epochs (265,068 / 265,074 / 265,072 vs 251,180 / 251,184 / 251,184), so
the two dataloaders are not sharding the dataset evenly. The loop
truncates to the `all_reduce(MIN)`, so it doesn't cause a boundary hang
by itself, but it is a sign that the data path is not symmetric.

### run_5 — single-NIC / bootstrap-on-QSFP56 (**failed with same signature**)

| | |
|---|---|
| Log | `job-driver-raysubmit_djwEAbn3maP2BWxp.log` (30 MB); rank 1 err `worker-020666...-05000000-132936.err`; flight recorder dumps `/data/nccl_dumps/nccl_trace_rank_{0,1}`; diagnostic tails `/data/diagnostic_train_logs/host_{0,1}_train_rank{1,0}.log` |
| Rank 0 host | 192.168.200.13 (`aitopatom-0512`), pid 53160 |
| Rank 1 host | 192.168.200.12 (`aitopatom-09a5`), pid 132936 |
| Started | 2026-04-10 01:19 |
| Crashed | 2026-04-11 07:10 |
| Runtime | ~30 h (well inside the 1.3-2.2 day failure window) |
| Stalled collective | `SeqNum=923553`, `NumelIn=7351296`, `OpType=ALLREDUCE`, first bucket of batch 210599 (sync=True) backward |
| Watchdog fires on | rank 1 (192.168.200.12) |
| Rank 1 state at dump | last enqueued 923,566 / last completed 923,552 (gap = 14, one DDP backward's worth of buckets minus the stuck one) |
| Rank 0 state in flight recorder | `last_enqueued == last_started == last_completed == 923552`; entry 923553 present in buffer with `time_created_ns` at 06:39:16.494 and `time_discovered_started_ns` at 07:09:17.247 — a **1801 s gap** ending exactly at comm abort |
| Desync debug | `[0] finished collective #923552, but didn't join collective #923553` |
| SIGSEGV lands on | rank 0 (192.168.200.13), same stack as all prior runs (`ncclLocalOpAppend → SaveProxy → … → autograd_hook → PythonEngine::thread_init`) |
| Timeline | 06:39:16.425 rank 0 `before_backward` for 210599 (sync=True); 06:39:16.494 rank 0 creates FR entry for 923553 (inside `pncclAllReduce`); 06:39:16.317 rank 1 `after_backward` (all 15 buckets already enqueued); 07:09:16.129 rank 1 watchdog fires; 07:09:17.247 rank 0's CUDA start-event for 923553 finally fires (comm abort); 07:10:16.388 rank 1 `before_optimizer` (`.item()` returns); 07:10:17.146 rank 1 watchdog kills the process; 07:10:23.578 rank 0 SIGSEGVs in `ncclLocalOpAppend` (+66 s from watchdog) |
| Per-epoch batches | rank 0 251,180 / rank 1 265,068-265,074 (same asymmetry as run 4) |
| NET | `Using [0]rocep1s0f1:1/RoCE [RO]; OOB enp1s0f1np1:192.168.200.13<0>` |
| Channels | 8 coll / 8 p2p |
| Mitigations vs run 4 | single RoCE HCA (`NCCL_IB_HCA=rocep1s0f1`); bootstrap moved off management LAN onto QSFP56 (`NCCL_SOCKET_IFNAME=enp1s0f1np1`); flight recorder persisted to `/data/nccl_dumps`; `TORCH_NCCL_DESYNC_DEBUG=1` |

Run 5 is the most important data point so far. **Both mitigations
from run 4 → run 5 were applied and neither fixed the bug:**

1. Dropped from dual-HCA to single-HCA — still failed.
2. Moved bootstrap TCP from 10.9.9.x management LAN to the QSFP56
   link — still failed.

And the flight recorder dump gives us **direct evidence** of what
prior runs could only infer:

- **Rank 0's FR entry for 923553 was created at `time_created_ns =
  1775914756494042183` (06:39:16.494) but its
  `time_discovered_started_ns = 1775916557247768642` (07:09:17.247)**.
  That is a 1,800,754 ms gap — the full 30-minute watchdog window.
  `time_discovered_started_ns` is set from the CUDA event recorded at
  kernel launch; the kernel for 923553 on rank 0 did not actually
  launch until the comm abort came in from the peer. This is the
  clean "stuck inside a single `pncclAllReduce` for 30 minutes"
  signal.
- **Rank 0's `pg_status` at dump time still shows
  `last_enqueued_collective == last_completed_collective == 923552`**,
  confirming the §3.2 argument: `lastEnqueuedSeq_` only bumps after
  `pncclAllReduce` returns, so a thread stuck inside `pncclAllReduce`
  leaves `last_enqueued` reporting the *previous* collective. The
  old "rank 1 never called `pncclAllReduce` for 1155874" reading of
  run 4 was definitively wrong.
- **Rank 1's FR entries 923564-923566 are in state `scheduled`**
  (not `started`); `pg_status` says `last_started_collective =
  923553`. That is consistent with rank 1 having kicked off the
  kernel for 923553 on its NCCL stream and then queued the next 13
  bucket allreduces behind it in-order, all blocked waiting for the
  peer — which is why rank 1's `.item()` blocks for 30 minutes via
  the cross-stream wait (§6.0).

(FR stack frames are empty for every entry — we did not set
`TORCH_NCCL_TRACE_CPP_STACK=1`, so the `frames` list is length 0.
The next run should set this so we get C++ stacks at collective
creation time.)

The rank 0 and rank 1 diagnostic tails pin the stall to the
first allreduce of the sync=True backward on batch 210599. Rank 1
finished that backward in 148 ms (06:39:16.169 → .317) and then sat
in `.item()` for exactly 31 minutes until the abort (07:10:16.388).
Rank 0 entered the same backward at 06:39:16.425 and never reached
`after_backward` — its autograd thread was parked inside
`pncclAllReduce` for the full window.

### Cross-run patterns

1. **SeqNum at failure is highly variable** — 629 K / 917 K / 923 K /
   1.16 M / 1.72 M across five failures. There is no "magic" collective
   count.
2. **The crashing host is not consistent** — both `.10` and `.11`, and
   both `.12` and `.13`, have hosted the SIGSEGV. Not a single-machine
   hardware fault.
3. **Which role crashes (rank 0 vs rank 1) is not consistent** — run
   0/1/2 crash on rank 0, run 4 crashes on rank 1, run 5 crashes on
   rank 0 again. Not a rank-specific code path.
4. **On the watchdog (peer) rank, `last_enqueued − last_completed` is
   always 14-23** — exactly one DDP backward pass' worth of gradient
   buckets. The peer rank finished its own backward (all buckets
   enqueued), then blocked in `.item()` via the compute-stream
   cross-stream wait (§6.0).
5. **Crash always lands in `ncclLocalOpAppend`**, always reached via
   the autograd hook dispatching the next bucket's allreduce, and
   always 60-70 s *after* the peer's watchdog fires. The autograd
   thread of the SIGSEGV'd rank is silent for the full 30 minutes
   before the watchdog and only resumes during teardown.
6. **The stall is inside a single `pncclAllReduce` call on the stuck
   rank.** Run 5's flight recorder entry for collective 923553 has
   `time_created_ns` at 06:39:16.494 and `time_discovered_started_ns`
   at 07:09:17.247 (a 1,801 s gap ending exactly at the peer abort).
   Run 5's `pg_status` at dump shows `last_enqueued == last_completed
   == 923552`, confirming that `lastEnqueuedSeq_` does not bump while
   the main thread is stuck inside the NCCL call. This is direct
   evidence, not inference.
7. **Every observed stall has landed on the *first* bucket allreduce
   of a `sync=True` backward pass.** Run 4: batch 75575 sync=True,
   first-bucket seq 1155874. Run 5: batch 210599 sync=True,
   first-bucket seq 923553. The stuck rank hits the stall on the
   first `pncclAllReduce` issued by the reducer once gradient
   accumulation exits `no_sync`. Still compatible with any "proxy
   pool gets wedged after many `no_sync` iterations accumulate
   state" hypothesis.

## 5. What we've ruled out

- **Thermals / PCIe power warnings.** Seen at init, not correlated with
  crash timing.
- **QSFP56 cable / single-port issue.** Failures have occurred on different
  physical machines and with different role assignments.
- **GDR.** Disabled by NVIDIA on DGX Spark by design; the "GPU Direct RDMA
  Disabled" log line is expected, not a misconfiguration.
- **Stack overflow of the NCCL background thread.** The historical
  `ulimit -s unlimited` paradox (glibc shrinks worker thread stack to 2 MB)
  was fixed upstream in NCCL 2.28; `/proc/<pid>/limits` shows
  `soft 8388608 / hard unlimited` on our workers.
- **A specific iteration / collective count.** See pattern 1 above.
- **A single rank or host.** See patterns 2-3 above.
- **End-of-epoch eval / checkpoint barrier.** The hangs land mid-epoch
  (run 4: 5 h 22 m into epoch 2, far from any `torch.distributed.barrier()`
  at epoch boundaries), so the evaluate/save-checkpoint/barrier sequence
  in `run_training` is not the stall site.
- **Application-level stall outside NCCL as the *cause*.** Earlier
  drafts read the run-4 rank-1 dump (`last_enqueued ==
  last_completed`) as proof that rank 1's main thread was outside
  NCCL during the 30-minute stall. That inference was wrong:
  `workEnqueue` (which bumps `lastEnqueuedSeq_`) runs *after*
  `pncclAllReduce` returns (PyTorch v2.8 `ProcessGroupNCCL.cpp`
  lines 3695 vs 3751; see §3.2). Run 5's flight recorder dump
  settles this directly — rank 0's entry for 923553 shows
  `time_discovered_started_ns − time_created_ns = 1801 s`, i.e.
  the kernel did not launch until the peer abort.
- **Second RoCE HCA as root cause.** Run 4 had both `rocep1s0f1`
  and `roceP2p1s0f1` active; run 5 pinned NCCL to `rocep1s0f1`
  only, and still failed with the same signature at ~30 h. Single
  HCA is not a fix, so dual-HCA traffic is not the cause.
- **Management-LAN bootstrap as root cause.** Run 4 used `enP7s7`
  (10.9.9.x) for NCCL bootstrap; run 5 used `enp1s0f1np1` (QSFP56,
  same link as the data path), and still failed. Bootstrap TCP on
  the management LAN is not the cause.

Three separate questions:

A. **Root cause** — why does the NCCL proxy op pool stop getting
   drained on R_stuck, so that R_stuck's autograd thread stalls inside
   `pncclAllReduce` → `ncclLocalOpAppend` for 30 minutes?
B. **Rank 0's symptom** — why does R_peer stall in `.item()` if it
   is not itself the rank whose NCCL path is wedged?
C. **SIGSEGV** — why does R_stuck die in `ncclLocalOpAppend` 60-70 s
   after the peer watchdog fires?

(B) is mechanical and fully explained by PyTorch internals, not a
separate bug (§6.0). (C) is a teardown-window race, not a steady-state
NCCL bug (§6.1). The real open question is (A), §6.2.

### 6.0 Why R_peer (rank 0) stalls in `.item()`

R_peer's diagnostic tail shows it stuck between `after_backward` and
`before_optimizer` — the only CUDA work in that window is
`accumulated_loss += loss.detach()` followed by `.item()`
(`trainable_architecture.py:160,163`). At the CUDA API level,
`.item()` does a `cudaMemcpyAsync` D2H + `cudaStreamSynchronize` on
the current compute stream; it does not, by itself, know anything
about NCCL. But the compute stream has already been made stream-
dependent on the NCCL stream during the *preceding* backward, by this
chain (PyTorch v2.8):

```
Reducer::finalize_backward()          reducer.cpp:1658
  └─ for each bucket:
       bucket.future_work->wait()     reducer.cpp:1683
  └─ ivalue::Future::wait()           ivalue_inl.h:896
       └─ synchronizeWithCurrentStreams()    ivalue_inl.h:1204
            └─ event.block(currentStream)   ≡ cudaStreamWaitEvent
```

Each DDP bucket's `future_work` carries an end-event recorded on the
NCCL stream; `event.block(currentStream)` inserts a
`cudaStreamWaitEvent` on the **compute** stream against that NCCL
end-event. So by the time backward returns, the compute stream has
~15 cross-stream waits queued against the NCCL stream for bucket
1155874..1155888. The next `cudaStreamSynchronize` on the compute
stream — the one inside `.item()` — must wait for all of those NCCL
end-events to fire, which cannot happen while R_stuck's allreduce
kernel for 1155874 is stuck waiting for its peer.

**GDR does not change this.** GDR affects whether the *data* moves
directly GPU↔NIC or is host-staged through the proxy thread; it does
not affect how the CUDA runtime coordinates the compute stream with
the NCCL stream. The `cudaStreamWaitEvent` is inserted by DDP
regardless. GDR-off only matters because it puts the proxy progress
thread on the critical path for every chunk of data, so any proxy
thread stall directly freezes the allreduce kernel (§6.2).

So R_peer is **not** stalled on its own. Its GPU is waiting,
transitively via the cross-stream event wait, for R_stuck to get past
`ncclLocalOpAppend` and actually run its half of the allreduce
kernel. When R_peer's 30-minute watchdog fires and aborts the comm,
the NCCL end-events complete with an error, the `cudaStreamWaitEvent`
unblocks, and `.item()` returns — which is exactly what the run-4
tail shows at 04:27:47 on rank 0.

### 6.1 Why the SIGSEGV lands in `ncclLocalOpAppend`

Once R_peer aborts its comm, R_stuck's `ncclLocalOpAppend` either
exits its free-list spin (because the proxy thread is now tearing
down and briefly refills the list) or progresses one step past it,
then dereferences pool state (`comm->proxyState->proxyOps`,
`pool->ops+opIndex`, `op->next`, `pool->ops[…].next`) that is being
freed on another thread. That is the fault.

The free-list spin at `proxy.cc:496`
(`while ((freeOp = pool->freeOps[tpLocalRank]) == -1) sched_yield();`)
is not itself the crash site — it cannot segfault on its own. The
crash is a pointer read in the enclosing function, and the stack is
repeatable only because there is exactly one way for an autograd
thread mid-`ncclLocalOpAppend` to race the peer's comm abort. This
makes the SIGSEGV a symptom of (A), not its own bug.

### 6.2 Why the proxy op pool stops being drained (the real root cause)

With GDR off, **the proxy progress thread is on the hot path for
every chunk of data moved** between nodes: the kernel stages data
into pinned host buffers, the proxy thread picks it up, posts the
RDMA send via `ibv_post_send`, polls the completion queue, and
refills the op pool slot. If the proxy thread stops making progress,
`ncclLocalOpAppend`'s slow-path spin on R_stuck *will* spin forever,
the allreduce kernel on R_peer never gets its peer data, and both
ranks go silent for exactly the 30-minute watchdog window. This is
the shape we see.

Plausible reasons for the proxy thread to stall, in rough order:

1. **ConnectX-7 / mlx5 driver stall on an RDMA completion.** The
   proxy thread is blocked in `ibv_poll_cq` (or a related mlx5 fast-
   path), waiting on a completion that never arrives because the
   driver lost track of a queue pair, got a silent ECN/CNP-induced
   reset, or hit a known mlx5 firmware corner case. Still the
   leading hypothesis because: (a) it cleanly explains why both
   ranks resume together only when the peer tears the comm down;
   (b) it is the kind of bug that manifests as "stalls after 1-2
   days of perfectly healthy traffic"; (c) run 5 ran with a single
   HCA and *still* failed at ~30 h, which only tightens this —
   a single stuck queue pair on the one active HCA is enough to
   wedge the whole allreduce. Distinguishing it from (2)/(3)
   requires a live py-spy / gdb sample of the proxy progress
   thread during a stall (§7.5).

2. **NCCL 2.28 proxy progress thread bug on aarch64 / Blackwell.**
   The DGX Spark is a new platform; NCCL 2.28 got the ulimit fix
   for aarch64 worker stacks but has not been soak-tested on GB10
   for days on end. A memory-ordering bug in `ncclLocalOpAppend` /
   `proxyProgressAsync` on a weakly-ordered aarch64 memory model
   could produce exactly this "pool drains fine for 10^6 ops then
   hangs" pattern. Harder to prove without an upstream repro.

3. **Host-staging buffer exhaustion / back-pressure deadlock.**
   With GDR off, NCCL uses a pinned host buffer per channel for
   data staging. If the proxy thread and the kernel-side enqueue
   end up in a state where the kernel is waiting for a staging
   slot that the proxy thread has marked "in flight" but never
   actually completed, the whole pipeline stops. This is a
   flavour of (1) but specifically in the NCCL-owned buffer
   accounting rather than the mlx5 layer.

4. **Teardown-race-style bug triggered by a transient proxy error
   that NCCL's error-handling path doesn't fully recover from.**
   I.e., the proxy thread hit a real error, marked a channel
   broken, but left the op pool's freelist in a state where
   `ncclLocalOpAppend` spins forever instead of returning
   `ncclSystemError`. Would need `NCCL_DEBUG=INFO` lines around
   the stall time to confirm.

5. **RoCE back-pressure / PFC misconfiguration** slowly building
   up queue depth until one side blocks indefinitely. Less likely
   because the traffic pattern is steady-state and we've been
   running 1-2 days without any congestion signal in the logs,
   but not ruled out.

What is cleanly ruled out as root cause:

- **Application code on R_stuck between collective N and N+1.**
  The diagnostic tail on rank 1 ends at `before_backward` for
  batch 75575 and never reaches `after_backward`, so the 30-min
  silence is entirely inside `loss.backward()`. DDP's reducer
  path inside backward is autograd-thread → `pncclAllReduce`; no
  application Python runs there.
- **MLflow / GIL contention, dataloader stall, `.item()` as a
  non-NCCL hang.** All of these would show up as gaps *between*
  backward passes. The rank-1 gap is inside backward.
- **DDP Reducer leaking buckets on R_stuck.** Reducer on R_stuck
  hasn't finished dispatching the current backward yet — there
  is nothing to leak.

## 7. Tests / experiments

### 7.1 Single RoCE HCA + QSFP56 bootstrap (run 5) — **FAILED**

Ran for ~30 h with `NCCL_IB_HCA=rocep1s0f1` and
`NCCL_SOCKET_IFNAME=enp1s0f1np1`. Crashed with the canonical
signature. Dual-HCA and management-LAN bootstrap are both ruled
out as root cause. The remaining relevant insight from this run
is the flight recorder evidence (§4 run_5) that the 30-minute
stall is inside a single `pncclAllReduce` call.

**For the next run**: add `TORCH_NCCL_TRACE_CPP_STACK=1` so
flight recorder entries carry their creation-time C++ stacks.
Without it, `frames` is an empty list in every entry and we lose
the most useful piece of evidence the recorder can produce.

### 7.2 Switch backend to gloo

`TorchConfig(backend=...)` already reads from yaml, so no code change —
just change the config file. Gloo is much slower but is a completely
separate communication path. If gloo also hangs at ~1 M collectives we are
looking at something above NCCL (DDP reducer, autograd engine, Python).
If gloo doesn't hang, NCCL / RoCE is the suspect.

### 7.3 Standalone `nccl-tests` soak

Run `nccl-tests/build/all_reduce_perf -b 8M -e 64M -f 2 -g 1 -n 10000000`
on the same pair of nodes, no PyTorch. If this also eventually stalls, we
have a pure NCCL / network reproducer to hand to NVIDIA. If it runs
indefinitely, the problem needs PyTorch / DDP in the loop.

### 7.4 OSU microbenchmarks — RDMA path without NCCL

Sits one layer below §7.3: OSU uses MPI over UCX/libibverbs, so it
exercises the exact same mlx5 driver and ConnectX-7 firmware that
NCCL's IB plugin goes through, but without any NCCL code on the hot
path. Combined with §7.3 this gives a clean isolation ladder:

| Test                  | NCCL | MPI/UCX | RDMA/mlx5 | PyTorch |
|-----------------------|:----:|:-------:|:---------:|:-------:|
| full training (§4)    |  ✓   |         |     ✓     |    ✓    |
| gloo backend (§7.2)   |      |         |           |    ✓    |
| `nccl-tests` (§7.3)   |  ✓   |         |     ✓     |         |
| OSU / perftest (§7.4) |      |    ✓    |     ✓     |         |

Build (aarch64, DGX Spark):

```bash
# Prerequisite: an MPI with UCX. openmpi is fine; whatever ships with
# the system is easiest. Verify UCX is wired in:
ompi_info | grep -i ucx          # should list btl / pml ucx components

# Grab OSU and build against that MPI:
wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.5.tar.gz
tar xf osu-micro-benchmarks-7.5.tar.gz
cd osu-micro-benchmarks-7.5
./configure CC=mpicc CXX=mpicxx --prefix=$PWD/install
make -j && make install
```

Two shapes of run to soak the driver path:

1. **Point-to-point bandwidth / latency** — the most direct RDMA
   stress test. If this stalls, the problem is below NCCL.
   ```bash
   mpirun -H host0,host1 -np 2 \
     -x UCX_NET_DEVICES=rocep1s0f1:1 \
     -x UCX_TLS=rc,ud,self \
     install/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw -i 100000 -x 1000
   mpirun ... install/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency -i 1000000
   ```
2. **Allreduce** — closer in shape to what training does, still no
   NCCL. Run with the same message sizes PyTorch's DDP is issuing
   (§4 run_5 stalled at `NumelIn=7351296` bf16 ≈ 14.7 MB):
   ```bash
   mpirun ... install/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce \
     -m 1048576:67108864 -i 100000 -x 1000
   ```

The iteration counts are deliberately high — the training stall
takes 20–30 h and ~1 M collectives to manifest, so a 30 s
microbenchmark run that passes tells us nothing. Plan on leaving
these running overnight at minimum, under the §8 hang watcher (set
`LOG_FILE` to wherever the benchmark's stdout lands).

Interpretation:

- **OSU stalls**: the bug is at or below the mlx5/verbs layer. Hand
  the reproducer + NIC counters + firmware version to Mellanox /
  NVIDIA networking. The PyTorch/NCCL side is just a victim.
- **OSU runs clean, `nccl-tests` stalls**: bug lives inside NCCL's
  use of the verbs layer (proxy scheduling, op pool, completion
  handling) — hypothesis §6.2 (2), (3), or (4).
- **OSU clean, `nccl-tests` clean, training hangs**: bug is in how
  PyTorch DDP drives NCCL (bucket sequencing, stream coupling,
  backpressure from the compute stream) rather than NCCL itself.

A bonus rung below OSU is the perftest suite (`ib_write_bw`,
`ib_send_bw`) which skips MPI entirely and pokes libibverbs
directly — worth running if OSU also looks suspicious and we want
to rule MPI/UCX out as a confounder.

### 7.5 Persistent flight-recorder dump to shared storage

Already set via `TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/data/nccl_dumps/nccl_trace_rank_`
(NFS-shared, reboot-safe). The *next* crash under this config should drop
per-rank JSON dumps with the last ~20 K collectives' metadata.

### 7.6 Live hang watcher (see §8) — highest priority

Auto-capture py-spy + gdb state when the training log goes stale. Now
that we know the stall is inside NCCL and specifically inside
`ncclLocalOpAppend`, the *key* thing a live sample would give us is
the state of the **NCCL proxy progress thread** (not the autograd
thread). A `py-spy dump --native --pid <pid>` on R_stuck, taken ~5
min into the stall, would show whether the proxy thread is:

- blocked in `ibv_poll_cq` / mlx5 poll — hypothesis §6.2 (1)
- spinning in NCCL proxy progress code with no pending work —
  hypothesis §6.2 (2) or (4)
- blocked in a futex / pool accounting path — hypothesis §6.2 (3)

Take the same sample on R_peer as well: its proxy thread should be
waiting for R_stuck's side of the allreduce, and any anomaly there
(e.g., proxy thread also wedged in `ibv_poll_cq`) supports a network-
layer stall over an NCCL-internal bug.

A `gdb -batch -ex 'thread apply all bt full'` capture is strictly
more informative than py-spy for this — take both.

### 7.7 Upstream bug search / report

- Collect run 4 artifacts (log, crash stack, NCCL version, topology) and
  cross-check against the NVIDIA DGX Spark / GB10 forum threads below.
- If run 5 reproduces, file against the same thread with the full C++
  backtrace and the flight-recorder dump — the existing forum thread only
  describes a bootstrap-time deadlock, which is a different bug.

## 8. Hang-watcher strategy

The watcher needs to be passive enough to run unattended for days, and
aggressive enough to capture state the moment a real hang happens. The
design:

1. **Signal: log file staleness + epoch marker.**
   `trainable_architecture.train_epoch()` writes per-batch debug lines
   to `/tmp/train_rank{rank}.log` (node-local) via
   `get_train_debug_logger`. A real hang is "no new line for
   `STALL_SECONDS` *and* last line is not `epoch_complete`". The
   `epoch_complete` marker suppresses false positives during rank 0's
   eval phase, when rank 1 legitimately produces no log output.
2. **Threshold well below the NCCL watchdog.** Default 5 minutes
   (`STALL_SECONDS=300`). NCCL's 30-minute watchdog is the hard
   upper bound — by the time it fires, the state we want has usually
   been torn down by the comm abort. 5 min gets us in well before
   that, while still giving real slow steps room to breathe.
3. **Two-sample discipline.** Always take two py-spy / gdb samples a
   few seconds apart. If the C++ frame at the top of the same thread
   is identical between the two, the thread is genuinely stuck; if
   it moves, it's just slow. Cheap to add, dramatically improves
   signal.
4. **Per-capture guard.** Once a capture fires on a given log file,
   do not re-fire on the same stall — otherwise the watcher spams
   the disk the whole 30-minute window. The next capture only arms
   once the log file starts advancing again.
5. **Run on both nodes.** Each watcher only cares about its local
   rank's log file. Let them fire independently. (The stuck rank
   and the peer both stop writing — stuck rank is in backward, peer
   is in `.item()` — so both watchers will trigger.)
6. **`py-spy --native` is required.** Without `--native`, py-spy
   only shows Python frames and misses every frame in the NCCL /
   CUDA / DDP reducer C++ layer. For this bug those are the frames
   we need.
7. **Don't send `SIGABRT` after capture.** It would kill the worker
   and potentially lose the natural NCCL watchdog path (and its
   flight-recorder dump). Let the hang persist after capture.

### 8.1 Preflights

The watcher is designed to be launched as root (`sudo bash
/data/hang-watcher.sh`) rather than relying on passwordless sudo for
individual tools. Before starting it, verify:

```bash
# Must be 0 for py-spy and gdb -p to attach (even for root, depending
# on kernel config — safer to just set it).
cat /proc/sys/kernel/yama/ptrace_scope
sudo sysctl -w kernel.yama.ptrace_scope=0   # if not 0

# gdb / gcore present on PATH:
which gdb gcore

# py-spy is NOT on PATH on these nodes — it lives in the ray-server
# uv venv. Point PY_SPY at the venv binary directly (NOT via `uv run`,
# which breaks under sudo because PATH gets scrubbed):
PY_SPY=/home/fadynakhla/ray-server/.venv/bin/py-spy
"$PY_SPY" --version

# Dump dir exists and is writable:
sudo mkdir -p /data/coredumps
```

Launch under sudo inside a tmux so it survives disconnects. The script
takes the rank it should watch as a positional arg — run it on each
node with the corresponding rank:

```bash
# on host 0 (rank 0)
tmux new -s hang-watcher
sudo bash /data/hang-watcher.sh 0
# C-b d to detach

# on host 1 (rank 1)
tmux new -s hang-watcher
sudo bash /data/hang-watcher.sh 1
```

Each invocation writes its artifacts under `/data/coredumps/rank${RANK}/`
so the two nodes don't clobber each other's `watcher.log`.

### 8.2 Watcher script

Drop this in `/data/hang-watcher.sh` on each node and launch it in a
tmux before starting the training job:

```bash
#!/usr/bin/env bash
# hang-watcher.sh — run in a tmux on each node.
# Usage: sudo bash hang-watcher.sh <rank>
# Alerts when the per-rank diagnostic log hasn't been written to for
# STALL_SECONDS, unless the last line marks epoch completion (in which
# case rank 0 is running eval and rank 1 is legitimately idle).
set -u

if [[ $# -lt 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: sudo bash $0 <rank>"
  exit 1
fi
RANK=$1

DUMP_ROOT=${DUMP_ROOT:-/data/coredumps}
DUMP_DIR=${DUMP_DIR:-${DUMP_ROOT}/rank${RANK}}
STALL_SECONDS=${STALL_SECONDS:-300}    # 5 min
POLL_SECONDS=${POLL_SECONDS:-30}
HEARTBEAT_SECONDS=${HEARTBEAT_SECONDS:-1200}   # 20 min
LOG_FILE=${LOG_FILE:-/tmp/train_rank${RANK}.log}
EPOCH_DONE_MARKER=${EPOCH_DONE_MARKER:-epoch_complete}
SECOND_SAMPLE_DELAY=${SECOND_SAMPLE_DELAY:-5}
# Absolute path to py-spy. On these nodes it lives in the ray-server uv venv
# rather than on PATH, so point directly at the venv binary (avoids
# `uv run` under sudo, which breaks because sudo scrubs PATH).
PY_SPY=${PY_SPY:-/home/fadynakhla/ray-server/.venv/bin/py-spy}

mkdir -p "$DUMP_DIR"

# Preflight — fail loud if we can't actually attach when we need to.
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: watcher must be run as root (e.g. 'sudo bash $0') — capture steps need CAP_SYS_PTRACE"
  exit 1
fi
# Yama ptrace_scope: root has CAP_SYS_PTRACE, which bypasses scopes 0-2.
# Only scope=3 blocks root, so that's the only value we need to guard on.
ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo "?")
if [[ "$ptrace_scope" == "3" ]]; then
  echo "ERROR: yama ptrace_scope=3 blocks ptrace even for root — py-spy/gdb will fail."
  echo "  Fix with: sysctl -w kernel.yama.ptrace_scope=1"
  exit 1
fi
"$PY_SPY" --version >/dev/null 2>&1 || { echo "ERROR: py-spy not runnable at $PY_SPY"; exit 1; }
command -v gdb    >/dev/null || { echo "ERROR: gdb not installed"; exit 1; }
command -v gcore  >/dev/null || echo "WARNING: gcore not installed — will skip core file capture"

# Timestamped log helper — writes to stdout (visible in tmux) and watcher.log.
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DUMP_DIR/watcher.log"
}

seen_mtime=0
stale_since=0
captured=0

capture_one() {
  local pid=$1 tag=$2 out_base=$3
  local out="${out_base}-pid${pid}-${tag}"
  # Python-side thread dump via faulthandler/SIGUSR1 (free; only useful
  # if faulthandler is registered in the training entrypoint).
  kill -USR1 "$pid" 2>/dev/null || true
  # Python + native C/C++ frames together — the key sample for an NCCL hang.
  "$PY_SPY" dump --native --pid "$pid" > "${out}.pyspy-native.txt" 2>&1 || true
  # Pure native stacks, all threads (proxy, watchdog, autograd workers).
  gdb -p "$pid" -batch \
    -ex "set pagination off" \
    -ex "info threads" \
    -ex "thread apply all bt" \
    > "${out}.gdb.txt" 2>&1 || true
  # Per-thread CPU — "spinning userspace" vs "blocked in syscall".
  top -H -b -n 1 -p "$pid" > "${out}.top.txt" 2>&1 || true
  # /proc snapshot — current syscall, kernel stack, state.
  {
    echo '# /proc/'"$pid"'/status'; cat /proc/$pid/status 2>/dev/null
    echo '# /proc/'"$pid"'/syscall'; cat /proc/$pid/syscall 2>/dev/null
    echo '# /proc/'"$pid"'/stack'; cat /proc/$pid/stack 2>/dev/null
  } > "${out}.proc.txt" 2>&1 || true
}

capture() {
  local reason=$1
  local ts host out_base
  ts=$(date +%Y%m%d-%H%M%S)
  host=$(hostname)
  out_base="$DUMP_DIR/hang-${host}-${ts}"
  log "HANG on $host: $reason"
  log "capture: writing artifacts to ${out_base}.*"

  nvidia-smi > "${out_base}.nvidia-smi.txt" 2>&1 || true
  ip -s link > "${out_base}.iplink.txt" 2>&1 || true
  # NIC counters — nonzero error/discard counters are a tell for RoCE issues.
  for nic in rocep1s0f1 roceP2p1s0f1 enp1s0f1np1; do
    ethtool -S "$nic" 2>/dev/null | grep -v ': 0$' \
      > "${out_base}.ethtool-${nic}.txt" 2>&1 || true
  done
  cp "$LOG_FILE" "$DUMP_DIR/" 2>/dev/null || true

  local pids
  pids=$(pgrep -f RayTrainWorker)
  if [[ -z "$pids" ]]; then
    log "capture: no RayTrainWorker pids found, skipping stack traces"
    return
  fi

  log "capture: attempting stack traces (sample 1) for pids: $(echo $pids | tr '\n' ' ')"
  for pid in $pids; do capture_one "$pid" sample1 "$out_base"; done
  log "capture: sleeping ${SECOND_SAMPLE_DELAY}s before second sample"
  sleep "$SECOND_SAMPLE_DELAY"
  log "capture: attempting stack traces (sample 2)"
  for pid in $pids; do capture_one "$pid" sample2 "$out_base"; done

  # Offline-analyzable core files — one per worker, once per capture.
  if command -v gcore >/dev/null; then
    log "capture: dumping core files with gcore"
    for pid in $pids; do
      gcore -o "${out_base}-core" "$pid" > "${out_base}-pid${pid}-gcore.log" 2>&1 || true
    done
  fi
  log "capture: done"
}

log "watcher starting on $(hostname) for rank ${RANK}: poll=${POLL_SECONDS}s stall=${STALL_SECONDS}s heartbeat=${HEARTBEAT_SECONDS}s log_file=${LOG_FILE} dump_dir=${DUMP_DIR}"
last_heartbeat=$(date +%s)
workers_were_up=0

while true; do
  if ! pgrep -f RayTrainWorker >/dev/null; then
    if [[ "$workers_were_up" -eq 1 ]]; then
      log "no RayTrainWorker processes running — resetting state"
      workers_were_up=0
    fi
    seen_mtime=0; stale_since=0; captured=0
    sleep "$POLL_SECONDS"; continue
  fi
  if [[ "$workers_were_up" -eq 0 ]]; then
    log "RayTrainWorker processes detected on $(hostname)"
    workers_were_up=1
  fi

  now=$(date +%s)

  if [[ ! -f "$LOG_FILE" ]]; then
    # Training might not have opened the log yet; just wait.
    sleep "$POLL_SECONDS"; continue
  fi

  # Periodic heartbeat for this rank's log file.
  if (( now - last_heartbeat >= HEARTBEAT_SECONDS )); then
    last_line=$(tail -n 1 "$LOG_FILE" 2>/dev/null || echo "")
    mtime=$(stat -c %Y "$LOG_FILE" 2>/dev/null || echo 0)
    age=$(( now - mtime ))
    if [[ "$last_line" == *"$EPOCH_DONE_MARKER"* ]]; then
      log "heartbeat: eval appears to be in progress (last write ${age}s ago). most recent log: $last_line"
    elif (( age >= 60 )); then
      log "heartbeat: log has been stale for ${age}s (capture fires at ${STALL_SECONDS}s). most recent log: $last_line"
    else
      log "heartbeat: training appears to be healthy (last write ${age}s ago). most recent log: $last_line"
    fi
    last_heartbeat=$now
  fi

  mtime=$(stat -c %Y "$LOG_FILE")
  if [[ "$mtime" -ne "$seen_mtime" ]]; then
    seen_mtime=$mtime
    stale_since=0
    captured=0
    sleep "$POLL_SECONDS"; continue
  fi

  if [[ "$stale_since" -eq 0 ]]; then
    stale_since=$now
    sleep "$POLL_SECONDS"; continue
  fi
  if (( now - stale_since < STALL_SECONDS )); then
    sleep "$POLL_SECONDS"; continue
  fi

  last_line=$(tail -n 1 "$LOG_FILE" 2>/dev/null || echo "")
  if [[ "$last_line" == *"$EPOCH_DONE_MARKER"* ]]; then
    sleep "$POLL_SECONDS"; continue
  fi

  if [[ "$captured" -eq 0 ]]; then
    capture "$LOG_FILE stale ${STALL_SECONDS}s, last line: $last_line"
    captured=1
  fi
  sleep "$POLL_SECONDS"
done
```

### 8.3 Optional: `faulthandler` / SIGUSR1 in the training entrypoint

For a Python-side thread dump without needing `sudo` or `py-spy`,
register `faulthandler` at the top of `ray_train.py`'s `main()`:

```python
import faulthandler, signal, sys
faulthandler.enable(file=sys.stderr, all_threads=True)
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
```

After that, `kill -USR1 <pid>` makes the process print every thread's
Python stack to stderr — the watcher already sends SIGUSR1 as part of
`capture_one`. It also catches `SIGSEGV`/`SIGABRT`/`SIGFPE` for free,
so the next crash will dump Python stacks before the process dies.

### 8.4 What to look for in a capture

The *most important* thread in the dump is NOT the one with the
familiar `ncclLocalOpAppend` / autograd_hook stack (that's the
symptom — we already know what its top frame looks like). It's the
**NCCL proxy progress thread**. Where it is parked tells us which
hypothesis in §6.2 is correct:

| Proxy thread is in... | Points to... |
|---|---|
| `ibv_poll_cq` / mlx5 ioctl, not moving between samples | (1) mlx5 driver stall on a lost RDMA completion |
| NCCL proxy progress loop spinning with no pending work | (2) NCCL 2.28 proxy thread bug |
| futex / condition variable on the op pool | (3) host-staging buffer back-pressure |
| NCCL error-handling path, but "stuck" | (4) transient proxy error + broken recovery |

The autograd thread on the stuck rank should show
`ncclLocalOpAppend → SaveProxy → … → pncclAllReduce → autograd_hook`
— same as every crash stack so far. The autograd thread on the peer
should show `cudaStreamSynchronize` inside `_local_scalar_dense_cuda`
via `accumulated_loss.item()` from
`trainable_architecture.py:163`. If either of those is different,
update §6.0 / §3.1 accordingly.

This aligns with the practices documented in stas00's `ml-engineering`
debugging section and the
[PyTorch troubleshooting docs for hangs](https://docs.pytorch.org/docs/stable/distributed.html),
which also recommend `TORCH_NCCL_DESYNC_DEBUG=1` (set) and capturing
via `faulthandler`/`SIGUSR1` for Python-level hangs.

## 9. References

- [Collective operations timeout on dual spark during distributed training](https://forums.developer.nvidia.com/t/collective-operations-timeout-on-dual-spark-during-distributed-training/366147) — our own forum thread
- [NCCL all-reduce deadlock on dual DGX Spark after successful channel establishment](https://forums.developer.nvidia.com/t/nccl-all-reduce-deadlock-on-dual-dgx-spark-after-successful-channel-establishment-affects-both-vllm-and-trt-llm/366127) — different bug (bootstrap-time, traced to Ubuntu 25.10 being unsupported)
- [DGX Spark NCCL Test: 15 GB/s so slow](https://forums.developer.nvidia.com/t/dgx-spark-nccl-test-15gb-s-so-slow/362446) — RoCE tuning baseline for DGX Spark
- [DGX Spark ↔ EdgeXpert NCCL only ~17 GB/s over 200 GbE](https://forums.developer.nvidia.com/t/dgx-spark-edgexpert-nccl-only-17-gb-s-over-200gbe/366055)
- [Connecting Two DGX Spark Systems via 200 Gb/s RoCE Network for Multi-Node GPU Training](https://medium.com/@dorangao/connecting-two-dgx-spark-systems-via-200gb-s-roce-network-for-multi-node-gpu-training-50d67d3630a5) — reference topology
- [PyTorch: Distributed communication package](https://docs.pytorch.org/docs/stable/distributed.html) — env vars, flight recorder, `TORCH_NCCL_DESYNC_DEBUG`
- [PyTorch issue #94393 — SIGSEGV when using DDP + NCCL + nsys profiling](https://github.com/pytorch/pytorch/issues/94393)
- [PyTorch issue #100240 — Signal 11 (SIGSEGV) with DDP + NCCL on A100](https://github.com/pytorch/pytorch/issues/100240)
- [PyTorch forum — Torch.compile + DDP SIGSEGV/SIGTERM during inference step](https://discuss.pytorch.org/t/torch-compile-ddp-sigsegv-sigterm-during-inference-step/203462)
- [stas00 / ml-engineering — debugging distributed training](https://github.com/stas00/ml-engineering)

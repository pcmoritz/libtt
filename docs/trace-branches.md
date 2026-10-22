# Conditional branches inside traces

A captured trace replays as one fixed command stream, so until now a trace
could only hold control flow whose outcome the host knew when it chose the
trace to replay. `third_party/tt_metal/trace_branch.patch` lets a trace branch
on a value the device computed, at the cost of one pipeline drain per branch.

## Device side

The prefetcher replays a trace with `CQ_PREFETCH_CMD_EXEC_BUF`, paging the
trace buffer from DRAM into its command queue and processing commands in
order. The patch adds `CQ_PREFETCH_CMD_EXEC_BUF_JUMP`, valid only inside an
exec_buf:

- `flags`: always, if the predicate is zero, or if it is nonzero; and whether
  the predicate is a 16-bit value;
- `base_addr`, `log2_page_size`: the interleaved DRAM buffer whose page 0
  holds the predicate (its first element);
- `target_offset`: the byte offset in the trace buffer to continue from.

A conditional jump reads 64 bytes of the predicate's page into a dedicated
L1 slot (`CommandQueueDeviceAddrType::EXEC_BUF_JUMP_SCRATCH`). The slot has
to be real L1: the prefetcher's stack and statics live in the RISC core's
local memory, which the NoC cannot write, and a read into them hangs the
dispatch pipeline. The read is tagged with the exec_buf's transaction id and
retired by id, and the slot is stamped beforehand and polled until the data
overwrites the stamp, because the id barrier alone returned before this
single read had landed.

A taken jump retires the pages read ahead, reseeks the exec_buf at the
target's page and continues at the offset inside it (`exec_buf_seek`).

## Host side

During capture, `TraceBranchBegin(cq, predicate)`, `TraceBranchElse(cq)` and
`TraceBranchEnd(cq)` record markers among the workloads. `record_end`
assembles the stream per device range in segments (prefix, arm A, arm B,
suffix) and emits, per branch:

    wait(all workers so far) + prefetch stall
    jump if zero -> else
    arm A, padded with dummy go signals
    jump -> end
    else: arm B, padded with dummy go signals
    end: wait(all workers)

The drain before the jump guarantees that the workload which produced the
predicate has finished. Two invariants make the device state identical after
either arm:

- The arms are padded to the same number of multicast and unicast launches,
  so the workers' launch-message read pointers and the completion counter
  agree at the join (every multicast launch completes on every tensix worker).
  Host bookkeeping is restored to the branch-point snapshot before arm B and
  checked for equality after it.
- Each segment is laid out in the kernel config ring as a trace of its own,
  with the prefetcher's binary cache reset, so no segment relies on an arm
  that may not have run; the join drain makes that assumption hold.

Commands appended between workloads go through the system memory manager's
reserve/write/push API like a workload's: the manager reserves space ahead
and tracks its own write offset, so appending to the end of its vector
corrupts the stream.

Branches do not nest, and loops are not covered yet (a backward jump).

## Test

`tests/tt_metal/test_trace_branch.cpp` captures traces with a branch between
single-core kernels that write and add constants into a DRAM page, with arms
of unequal length, workloads before and after the branch, a branch that reads
a predicate no traced workload wrote and one that ends the trace, and replays
them with alternating predicate values.

## Up the stack

`tt_mlir_trace_branch.patch` wires the mechanism into compiled programs:

- The trace hoist keeps a two-way `ttcore.case` with a device-resident
  selector inside the trace instead of rejecting it (a host-resident selector
  still becomes a trace key, which costs no drain). A device-selected case
  may not hold another one in its branches, since trace branches do not nest.
- The runtime marks the thread while a trace's run-and-capture program
  executes. In that state the case handler (`tt_mlir_control_flow_runtime.cpp`)
  runs both branch programs, between `TraceBranchBegin/Else/End` when the
  queue is capturing, and copies each branch's results into one result
  buffer allocated from the first branch's outputs, so the case has a single
  result whichever branch replays. The uncaptured warmup takes the same path,
  which is what compiles both branches. The predicate is the selector's first
  element, moved to interleaved DRAM (and widened to 32 bits for narrow
  integer types) if needed, as captured ops.

The branch programs must be free of side effects the other branch would not
perform, since both execute during capture.

`tests/libtt/traced_device_branch_regression_test.py` checks a conditional
on a value computed from the data, branches of unequal size, and a device
predicate combined with a host predicate, and reads the exported IR to show
the function stays one trace.

## Not covered yet

Nested device-selected conditionals, cases with more than two branches, and
loops with device-dependent trip counts (a backward jump).

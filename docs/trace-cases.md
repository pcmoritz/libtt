# Conditionals inside traces

A captured trace replays one fixed command stream, so a conditional used to
run eagerly between traces: the runtime read its index on host and ran the
branch's program, with traces of its own. For an index the device computes,
the read waited for all the work queued before it, and every value crossing a
trace boundary was copied. A traced function now keeps its conditionals inside
its trace, and the device picks the branch.

## Device side

`third_party/tt_metal/trace_case.patch` adds one prefetcher command,
`CQ_PREFETCH_CMD_EXEC_BUF_SKIP`, valid inside an exec_buf. It skips the next
`bytes` bytes of the trace, always or depending on a device value: the low
`value_bits` bits of the first word of page 0 of an interleaved DRAM buffer,
compared with `value`. The prefetcher reads the word into the command's own
slot in its command buffer, and skips by consuming the bytes through the same
refill loop that relays commands spanning refills. Skipped commands launch
nothing and fetch none of the data they would relay; their bytes are still read
from DRAM.

During capture, `MeshCommandQueue::record_case_begin(index, index_bits)`,
`record_case_branch(value, equal)` and `record_case_end()` record markers
among the workloads. `record_end` lays out a case as

    boundary + prefetcher stall
    skip if branch 0's test fails -> b1; branch 0; boundary; skip -> end
    b1: skip if branch 1's test fails -> b2; branch 1; boundary; skip -> end
    ...
    last branch (taken when no test passes); boundary
    end:

A boundary returns the device to the state a replay starts in, with the
sequence a replay issues before its trace (`reset_worker_dispatch_state_on_device`):
it waits for the programs since the last boundary, resets the workers' launch
message read pointers and clears the completion counters. The stall before a
case makes the prefetcher wait until the dispatcher gets there, so the index it
reads is the one the drained programs wrote. Each run of workloads between
markers is laid out as a trace of its own (kernel config ring, prefetcher
cache), so no command depends on which branch led to it. Cases nest; a case
costs two drains of the device.

Only the workloads after the last case can still run when the trace ends, so
only they are equalized across device ranges with dummy go signals and
described by the trace's worker descriptors. A sub-device that runs programs
before the last case but not after it gets one dummy go signal there, so a
replay resets its launch state first.

`tests/tt_metal/test_trace_case.cpp` replays traces with two-way and many-way
cases, empty and nested branches, cases at both ends of a trace, an index a
slow kernel writes just before the case, branches larger than the prefetcher's
command buffer, masked indices and branches recorded out of order.

## Up the stack

`third_party/tt_xla/tt_mlir_trace_case.patch` and
`tt_mlir_control_flow_runtime.cpp`:

- The trace hoist keeps a `ttcore.case` in the trace when its branch programs
  replay from a trace too. Its index may be a device value or an argument the
  caller passes from host, which the trace stages into a DRAM slot like any
  other host input.
- During a trace's capture program (the uncaptured warmup, the captured run and
  the first replay), the case handler runs every branch, between the markers
  while the queue records, and leaves their results in the same buffers. The
  warmup compiles every branch and reads the index on host, and the capture
  lays out the branch it took first: that branch's results become the case's,
  so the replays that take it copy nothing. The other branches copy their
  results into them.
- A result a branch passes through from an input the case is the last to read
  takes over that input's buffer: the hoist makes the other branches copy their
  result into the input and return it, and no branch copies anything for the
  one that passes it through. The capture program writes host-staged inputs
  again after the warmup, which may have written into a donated slot.
- The index is tested in place where its type allows (32-bit integers, and the
  bits of a floating-point predicate but its sign); other types are converted
  first.

Since the warmup runs every branch, a case stays out of the trace if a branch
(or a loop body it runs) writes into anything but its own values and the input
it was donated, such as a cache it updates in place; it then runs eagerly as
before. Loops whose trip count the device decides still run eagerly too.

`tests/libtt/traced_device_case_regression_test.py` checks conditionals on
computed values, through counted loops, results passed through from dead and
live inputs, that the function stays one trace, and that the call returns
before the device finishes the work queued ahead of it.

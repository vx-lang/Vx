# The litmus tests on a GPU

These programs run litmus tests from `../tests/` on a GPU, written in Vx. herd7 checks what the
model allows. These programs check what the PTX that vxc makes does on real hardware.

Each program runs 2 blocks of 1 lane, and repeats the test for 100000 rounds in one launch. A
handshake on `go` starts each round in both blocks at about the same time. The program prints
how many rounds gave the result in question.

| Program | Litmus test | The model says |
| --- | --- | --- |
| `mp_rel_acq_blocks.vx` | `message_passing_release_acquire` | never: must print 0 |
| `sb_seq_cst_blocks.vx` | `store_buffering_seq_cst` | never: must print 0 |
| `lb_relaxed_blocks.vx` | `load_buffering_relaxed` | never: must print 0 |
| `mp_relaxed_blocks.vx` | `message_passing_relaxed_flag` | sometimes: a control |
| `sb_rel_acq_blocks.vx` | `store_buffering_release_acquire` | sometimes: a control |

A control shows whether the GPU gives weak results at all in this setup. If no control prints
more than 0, a 0 from the other programs is weak evidence.

Each `_stress.vx` program is the same test with 14 more blocks. They write and read their own
part of a buffer while the test runs, which makes weak results more likely.

`lb_relaxed_blocks.vx` is the one most likely to fail. vxc makes a relaxed load followed by a
relaxed store (`ld.relaxed`, then `st.relaxed`), and the PTX model does not order these two.

## Running them

They need a GPU. On a CPU the blocks run one after the other, and the first round waits forever.

```bash
vxc formal/memory_model/gpu/mp_rel_acq_blocks.vx --host default --machine fleet/h100-sxm.vx -o mp
for i in $(seq 10); do timeout 120 ./mp; done
```

`tests/optimizations/pass/atomic_orders_and_fence_scopes_in_ptx.vx` checks the PTX of each atomic
order and fence scope, without a GPU.

## Not covered

- The async copy tests: `raw::async_copy` is a plain copy today, so there is no copy engine to
  test.
- Atomics of block scope: every atomic in `core::sync::atomic` has system scope.

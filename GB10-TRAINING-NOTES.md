# Field notes: training a DeepSpec draft on DGX Spark (GB10)

The data-generation notes are in [`TWO-SPARK-NOTES.md`](TWO-SPARK-NOTES.md). This file is about
the **draft-training** step itself — what breaks when you run DeepSpec's trainer on GB10's ARM +
unified-memory platform, and the fixes that got a DSpark and an EAGLE-3 draft trained end-to-end.

Everything here was hit and fixed on a real 2×GB10 run. None of it is Nemotron-specific — it
applies to any DeepSpec draft (Qwen3 / Gemma / Nemotron) trained on this hardware.

---

## A. The big one: DeepSpec's draft attention is `flex_attention`-native, and `flex_attention` doesn't work on GB10

This is the wall you hit first, and it blocks *all* draft flavors.

DeepSpec builds its DSpark and EAGLE-3 attention masks as **FlexAttention `BlockMask`** objects and
dispatches through `torch.nn.attention.flex_attention`. On a GB10 that gives you three dead ends:

1. **Uncompiled `flex_attention` is pathologically slow.** Expect ~0 optimizer steps in 20 minutes,
   with the GPU sitting in a low-power busy-wait (~37 W at full clocks). Not viable.
2. **`flex_attention` + `torch.compile` crashes.** Inductor on ARM is immature enough that the
   compile path raises `InductorError` on this platform. (This may improve with newer PyTorch — test
   it before assuming; if compile works for you, prefer it.)
3. **`sdpa` rejects the mask.** `scaled_dot_product_attention` wants a dense tensor and errors on the
   `BlockMask`: `attn_mask must be a Tensor, not BlockMask`.

### The fix: a dense-tensor mask that materializes the *identical* predicate for `sdpa`

The mask predicate DeepSpec expresses as a flex `mask_mod` is just a boolean function of
`(batch, query_idx, key_idx)`. You can materialize the exact same predicate into a dense
`[B, 1, Q, KV]` boolean tensor (`True` = attend) and hand that to `sdpa` — same math, no flex, no
compile. Add a `dense=` path to the mask builder and select it whenever the draft isn't actually
running `flex_attention`, and set the trainer's attention implementation to `sdpa`.

Sketch for the DSpark mask (mirror your `mask_mod` exactly — this is illustrative, not a drop-in):

```python
def create_dspark_attention_mask(*, anchor_positions, block_keep_mask, seq_len,
                                 block_size, device, dense=False):
    if dense:
        # Same predicate as the flex mask_mod, as a [B, 1, Q, KV] bool tensor (True = attend).
        bsz, num_blocks = anchor_positions.shape
        q_len  = num_blocks * block_size
        kv_len = seq_len + q_len
        q_idx  = torch.arange(q_len,  device=device)
        kv_idx = torch.arange(kv_len, device=device)
        q_block = q_idx // block_size
        anchor  = anchor_positions.to(device)[:, q_block]                       # [B, Q]
        is_ctx   = (kv_idx < seq_len).view(1, 1, kv_len)
        mask_ctx = is_ctx & (kv_idx.view(1, 1, kv_len) < anchor.unsqueeze(-1))
        kv_block = torch.clamp(kv_idx - seq_len, min=0) // block_size
        is_draft   = (kv_idx >= seq_len).view(1, 1, kv_len)
        mask_draft = is_draft & (q_block.view(1, q_len, 1) == kv_block.view(1, 1, kv_len))
        valid = block_keep_mask.to(device=device, dtype=torch.bool)[:, q_block] # [B, Q]
        mask  = (mask_ctx | mask_draft) & valid.unsqueeze(-1)
        return mask.unsqueeze(1)                                                # [B, 1, Q, KV]
    ...  # original flex path unchanged
```

Then at the call site: `dense=(self.config._attn_implementation != "flex_attention")`, and set
`TRAIN_ATTN_IMPLEMENTATION = "sdpa"` in the draft-config builder.

**Verification that matters:** write a tiny CPU test that builds both the flex `BlockMask` and your
dense tensor for random shapes and asserts they encode the same attend/no-attend decisions. Do not
trust the dense mask until it passes bit-for-bit against the flex predicate on several shapes.

After this, steps run on real compute (power cycles 40–67 W instead of the 37 W busy-wait) with zero
errors.

### EAGLE-3 has a silent-corruption trap here — read this before you train one

EAGLE-3's modeling already has a non-flex fallback (`prepare_4d_causal_attention_mask`), so it *looks*
like you don't need the dense fix. **You do.** That fallback is a **plain causal mask**, which is
correct only for `ttt_length == 1`. For any `ttt_length > 1` (the whole point of EAGLE-3), it drops
the **suffix-diagonal** term of the real `eagle3_mask_mod` predicate — the draft trains happily, loss
looks healthy, and you ship a **quietly degraded** model because attention leaked across the
train-time-test segments.

So for EAGLE-3, apply the same dense-materialization treatment to `create_eagle3_attention_mask`
(reproducing `causal | suffix` from `eagle3_mask_mod`), and route the non-flex call sites to it
instead of the plain-causal fallback. One extra subtlety: flex emits **zeros** for fully-masked query
rows, but `sdpa` emits **NaNs** for an all-`False` row — so for any empty row, unmask that row's own
key slot as a guard (those rows carry no loss, so it's harmless). Same CPU bit-equivalence test
applies.

---

## B. Step-time decay on unified memory (and how to stop it)

Once training runs, you may see **step time climb over the run** — e.g. 20 → 35 → 60+ min per step —
with the GPU in the same low-power busy-wait and the box's used memory ratcheting from a few GB toward
the full 128 GB. It is **not** thermal (clocks stay pinned, no throttle flags) and **not** a Python
leak (RSS is flat).

Two compounding causes, both fixable:

1. **The CUDA caching allocator never returns freed blocks to the OS.** On a discrete-VRAM box that's
   invisible; on GB10's *unified* pool it means the allocator ratchets toward the whole 128 GB and
   starves the OS page cache and the dataloader. `expandable_segments:True` curbs fragmentation growth
   but does **not** release memory. The reliable per-step release is an explicit
   `torch.cuda.synchronize(); torch.cuda.empty_cache()` at the optimizer-step boundary (milliseconds
   against minute-scale steps).
2. **Per-micro-batch device syncs** amplify it. Profiling (py-spy on the *child* trainer process, not
   the `torch.multiprocessing.spawn` supervisor at PID 1) showed the main thread pinned on:
   - a `torch.tensor(scalar, device="cuda")` built *inside* a `torch.where` on the hot path — one
     H2D copy + stream sync per micro-batch (×128+ per step). Pass the Python scalar directly; modern
     `torch.where` folds it into the kernel with no allocation.
   - an `if some_tensor.item() > 0:` branch — a full stream stall. When the guarded compute is
     value- and gradient-identical either way (e.g. a `num / (den + eps)` where `den == 0` already
     yields 0), just compute it unconditionally and delete the branch.

   Each such sync makes a step's wall time a function of driver/allocator latency, which is exactly
   what degrades as the pool tightens.

**Numerics gate:** whenever you touch the hot path, assert that per-step losses stay **byte-identical**
to a pre-patch reference run (the data order is deterministic, so they should). That's how you know a
"performance" change didn't move the math.

**If decay still persists** (it can, on a long run), the pragmatic answer is to **restart-cycle**:
checkpoint every N steps and, when step time crosses ~2× your fresh-process baseline, stop → drop
caches → relaunch. DeepSpec's `discover_latest_checkpoint` auto-resume makes this exact
(`next_micro_step` is preserved), so you lose at most N steps. A small node-side watchdog that does
this on a fixed step cadence keeps the whole run in the fast regime unattended. A **full OS reboot**
clears the accumulated state completely — worth doing between long runs.

---

## C. Checkpoint every few steps, not every few thousand

The stock configs set `checkpointing_steps=3000`, which assumes a run of many thousands of steps. A
small proof run (tens of steps) then writes **no** mid-run checkpoint and is all-or-nothing against
the decay/restart cycle above. Set `logging.checkpointing_steps` low (1–5) for short runs so a restart
costs at most a few steps.

If you train **data-parallel across two boxes with separate local disks** (no shared filesystem),
note that DeepSpec's checkpoint save gates directory creation, the model-weight write, and the
`step_latest` symlink to the **global-main rank only** — which leaves every other rank's local disk
without the checkpoint and crashes the instant a non-rank-0 process tries to write its own state file.
Under `sharding_strategy="no_shard"` every rank already holds the full state, so the fix is to let
**each** rank create its own directory, write its own copy, and update its own `step_latest`. (This is
`no_shard`-only reasoning; a genuinely sharded strategy needs real re-sharding, not a per-rank copy.)

---

## D. Benchmarking the trained draft: two harness landmines

When you run DeepSpec's acceptance eval against a **Nemotron-H** target:

1. **`logits_to_keep` isn't implemented on the vendored hybrid model.** The eval calls the target with
   `logits_to_keep=1` expecting a `[1, 1, vocab]` slice; the vendored `modeling_nemotron_h.py` swallows
   the kwarg and returns logits for **every** input position, so the eval crashes on a shape mismatch.
   Add the standard HF handling — slice `hidden_states[:, -logits_to_keep:, :]` before the LM head,
   with `0` meaning "keep all" so default behavior is unchanged.
2. **The generic decode cache falls back to O(n²) recompute.** The eval's manual generation loop builds
   a generic `DynamicCache`, but Nemotron-H's Mamba/SSM layers need their own hybrid cache. Without it
   the model logs a warning and recomputes the **whole** sequence per generated token — correct output,
   but quadratic, so a full-scale eval takes days. Until the eval loop is wired to the hybrid cache,
   use **small sample counts** for a directional read (the acceptance number converges fast and is
   consistent across tasks), or measure acceptance through a serving engine that already implements the
   hybrid cache correctly.

---

## E. A few smaller gotchas

- **`--opts` takes one key=value per flag.** Repeat the flag (`--opts a=1 --opts b=2`), don't pass two
  values after one flag.
- **Config field naming drift.** A hybrid target may name its norm epsilon `norm_eps` rather than
  `rms_norm_eps`; alias it in the draft-config builder or the build asserts on a missing field.
- **Cap container memory** so a bad allocation kills the container, not the box:
  `--memory=105g --memory-swap=105g` on a 128 GB GB10 leaves headroom for the OS. Without it, a large
  target load can wedge the machine hard enough to need a power cycle.
- **`py-spy` works on GB10 ARM** (install it in-container; reinstall after any container recreate).
  Profile the trainer **child** process, not container PID 1 — PID 1 is the spawn supervisor sitting in
  an idle `select` loop.

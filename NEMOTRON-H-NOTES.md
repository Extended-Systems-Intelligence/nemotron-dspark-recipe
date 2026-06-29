# Nemotron-H field notes: hybrid-architecture landmines

Practical notes on the **Nemotron-H hybrid Mamba-Transformer** checkpoints, focused on the
places where stock Hugging Face code paths don't behave the way speculative-decoding data-prep
(and general fine-tuning) expect. These are **plumbing workarounds for upstream interactions**,
observed on a vendored Nemotron-H checkpoint — class names, line numbers, and exact failure
text are version-specific, so verify against your `transformers` version and checkpoint before
relying on any specific detail.

The reason any of this matters: DSpark/EAGLE-style data-prep runs the **target** model forward
to capture hidden states. On a pure-transformer target that "just works." On a hybrid target,
the generation/cache machinery is a different, less-exercised code path — and that's where it
breaks.

---

## A. Notes that directly affect the DSpark recipe (target forward / hidden-state extraction)

These bite the **target-cache** stage of [`GUIDE.md`](./GUIDE.md) §2.

1. **Don't rely on `generate()` to drive the forward pass.** On the hybrid checkpoints, the
   generation glue can be inconsistent: `prepare_inputs_for_generation` may hand back a
   `past_key_values` object while the model's `forward` expects `cache_params`, and
   `cache_position` can arrive as `None`, producing a `TypeError` mid-decode. For
   hidden-state extraction you don't need generation anyway — run plain forward passes.

2. **Run the forward with cache disabled.** Set `use_cache=False` (equivalently, drive the
   model with `cache_params=None`) so you take the no-cache recompute path rather than the
   hybrid dynamic-cache path. This is the path that training/forward exercises reliably.

3. **The hybrid dynamic cache is the fragile part.** The combined attention+SSM dynamic-cache
   object is easy to mis-construct (e.g. a missing `conv_kernel_size`), and some of its state
   updates assume a tensor where a *list* is actually present (calling `.device` on a list).
   Avoiding the cache entirely (point 2) sidesteps all of this.

4. **Cost of the workaround.** No-cache forward is full recompute — `O(n^2)` in sequence
   length — so it is slower than a cached decode. For cache-generation over a fixed dataset
   this is fine (you're doing one forward per sample, not autoregressive decode), but keep
   sequence lengths reasonable and batch sensibly.

5. **Request hidden states explicitly.** Pass `output_hidden_states=True` and read the
   per-layer hidden states for your `target_layer_ids`. The returned tuple includes a hidden
   state for **every** decoder layer, Mamba (SSM) layers included — so your tap indices index
   into the full hybrid stack, not just the attention layers. Choose a low/mid/high spread; the
   attention layers tend to carry the richest cross-token signal, so taps on or adjacent to
   them are a sensible default until you have ablation data.

6. **For production serving, use a purpose-built engine.** The reliable way to *serve*
   Nemotron-H at speed is a dedicated inference engine (e.g. vLLM / TensorRT-LLM) rather than
   raw `transformers` generation. The no-cache recompute path above is for **data-prep**, not
   for latency-sensitive serving.

---

## B. General Nemotron-H training plumbing (useful if you also fine-tune the target)

Not required for draft training, but commonly hit when working with Nemotron-H under the
PEFT / TRL stack. All generic upstream-interaction notes.

1. **Prefer BF16 base weights; avoid bitsandbytes 4-bit.** 4-bit (NF4) quantization can leave
   the parameter as a packed weight blob that a PEFT matmul then tries to multiply directly,
   yielding a `mat1 x mat2` shape error. The Mamba/SSM layers stay in BF16 regardless, so the
   memory saving from 4-bit is smaller than for a pure transformer — BF16 is the path of least
   resistance.

2. **Pin the loss path that doesn't assume KV cache.** TRL's default chunked cross-entropy
   reads `outputs.past_key_values`, which a hybrid output object may not expose, raising an
   `AttributeError` inside the chunked-CE forward. Selecting the plain negative-log-likelihood
   loss (`SFTConfig(loss_type="nll")`) avoids the chunked path.

3. **`use_cache=False` during training**, for the same hybrid-cache reasons as §A.

4. **Beware `device_map="auto"` on unified-memory / ARM hosts.** On platforms where the
   accelerator reports free memory as `N/A`, `device_map="auto"` can mis-plan and try to
   CPU-offload. Pin placement explicitly, e.g. `device_map={"": 0}`.

---

## C. Choosing `target_layer_ids` for a Nemotron checkpoint

There is no single right answer; it depends on layer count and the hybrid layout. First
confirm what you're indexing into: for Nemotron-H, `num_hidden_layers` counts **every** layer
in the hybrid schedule (Mamba/SSM, attention, and MLP layers alike), and `output_hidden_states`
returns one hidden state per layer in that schedule (plus the embedding output). So your tap
indices address the *full* hybrid stack, not an attention-only subset — which means you must
know which positions in the schedule are attention layers before choosing taps. Inspect the
model's config (e.g. the hybrid/override pattern) to map that out. A reasonable starting
heuristic, once you know the layout:

- Read `num_hidden_layers` from the target config and confirm it against the actual hidden-state
  tuple length returned by a forward pass.
- Pick ~5 taps spread from early to late (e.g. roughly the 10%, 35%, 60%, 85%, and final-ish
  depths), matching the count DeepSpec's own configs use.
- Bias taps toward attention layers where the hybrid schedule places them.
- Treat the choice as a tunable: re-run a small training + eval with a couple of tap sets and
  keep the one with the best accepted length. If you find a good set for a public Nemotron
  checkpoint, please contribute it back.

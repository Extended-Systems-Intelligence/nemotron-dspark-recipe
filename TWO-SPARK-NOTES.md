# Field notes: two DGX Sparks (GB10) for the data-generation step

Practical notes from running this recipe's **data-generation** stage across a pair of NVIDIA
**GB10 DGX Spark** boxes. Read this alongside [`GUIDE.md`](./GUIDE.md) §1–2 and
[`NEMOTRON-H-NOTES.md`](./NEMOTRON-H-NOTES.md). Everything here is generic infrastructure — no
model changes — and is offered as one working setup, not the only one.

## What two Sparks actually buys you (and what it doesn't)

Be clear-eyed about where the parallelism helps:

- **It parallelizes answer regeneration (GUIDE §1.2)** — the step that serves the target and
  regenerates answers over your prompt set. That step is throughput-bound on the target server,
  so a second Spark serving a second copy of the target roughly doubles it. This is the win.
- **It does *not* give you distributed draft training here.** The target-cache build (§2) and
  the draft training (§3) in this recipe run **single-node**. The draft is small; the cache
  build is a single-GPU forward loop. A second Spark sitting idle during those stages is normal.
- **A large (100B-class) target does not fit tensor-parallel across two GB10s in BF16** — see
  the sizing note at the bottom. For a big target you serve the **NVFP4** build on one Spark,
  not BF16 split across two. For the Nano-class target in this recipe, one Spark serves it fine.

So: two Sparks = ~2× faster data-gen, same single-node training. Size your expectations there.

## The setup: two target servers behind a least-connections load balancer

The data-gen client (`generate_train_data.py`) accepts multiple `--server-address` values, but
in practice it does not spread load evenly across them under variable-length prompts — one
server ends up saturated while the other idles, and if one server hiccups the client can stall
waiting on it. The robust fix is to take balancing away from the client entirely:

1. **Serve the target on each Spark** as an OpenAI-compatible endpoint (we used vLLM with the
   **NVFP4** target build — it fits comfortably and is fast on Blackwell; see the GB10 notes
   below for why NVFP4 over BF16 for *serving*). Same `--served-model-name` on both.
2. **Front them with a least-connections reverse proxy** on one box, and point the client at the
   proxy as a *single* server address. Least-connections sends each request to whichever server
   has the fewest in-flight requests, which self-balances even when prompt lengths vary, and it
   ejects a dead/slow server and retries the other automatically.

A minimal nginx config (adjust addresses/ports to your two Sparks):

```nginx
events { worker_connections 8192; }
http {
  upstream targets {
    least_conn;
    server 127.0.0.1:8010     max_fails=1 fail_timeout=15s;  # target on this box
    server SPARK_B_IP:8010     max_fails=1 fail_timeout=30s;  # target on the other box
  }
  server {
    listen 8020;
    location / {
      proxy_pass http://targets;
      proxy_connect_timeout 5s;
      proxy_read_timeout 60s;              # a wedged server that connects but never streams
      proxy_next_upstream error timeout http_500 http_502 http_503 http_504;
      proxy_next_upstream_tries 2;         # ...times out and the request retries the other one
      proxy_buffering off;
    }
  }
}
```

Then run the data-gen client against the proxy only — e.g. `--server-address 127.0.0.1:8020` —
and raise `--concurrency` enough to keep both targets' batches full (both servers at ~60–120
concurrent requests each is a reasonable target; watch each server's engine log for its running
count and back off if the client itself becomes the bottleneck).

## GB10-specific gotchas

- **Prefer the NVFP4 target build for *serving*.** On a GB10 the usable CPU-side staging RAM is
  a fraction of the unified pool, so loading a large BF16 checkpoint into the inference engine
  can be unreliable, and quantize-at-load (`--quantization fp8`) reads the full BF16 *and* builds
  the quantized copy at once, which can thrash the box into swap. A prebuilt **NVFP4** checkpoint
  loads within staging RAM and is Blackwell-native. (This is about the *serving* engine only —
  the target-cache build in §2 loads BF16 weights via `transformers`, which is a different path.)

- **Unified-memory allocations can get *stuck* — and only a reboot clears them.** After heavy or
  repeated inference workloads, a GB10 can reach a state where new CUDA allocations fail with
  `NVRM: ... Out of memory [NV_ERR_NO_MEMORY]` in `dmesg` even though the box looks idle and
  `free` shows plenty of RAM. The server process is up but any real generation hangs. Stopping
  the container and dropping caches (`echo 3 > /proc/sys/vm/drop_caches`) does **not** always
  reclaim it; a **reboot** does, and afterward the server both stops erroring *and* runs at full
  clock/throughput (a half-degraded GPU can quietly serve at a fraction of its real speed). If
  one Spark is mysteriously slow or flaky, check `dmesg` for `NV_ERR_NO_MEMORY` before assuming
  it's a config problem — and check `nvidia-smi --query-gpu=clocks_throttle_reasons.active` to
  rule thermal in or out (a stuck-memory GPU runs *cold and idle*, the opposite of thermal
  throttling).

- **A wedged inference server still answers `/v1/models`.** The failure mode above presents as:
  the HTTP API responds to metadata requests (so a naive health check passes) but a real
  completion request hangs indefinitely. If you script auto-recovery, probe an actual short
  generation with a hard timeout — not just `/v1/models` — and only restart on a hang. Gate the
  restart on "API up **and** generation hangs"; if the API itself is down the server is merely
  loading, and restarting it mid-load just thrashes.

- **`drop_caches` before standing up a new model.** On the unified-memory platform, page cache
  counts against what CUDA can grab; `sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'` before
  launching a server avoids a spurious first-allocation OOM.

## Sizing note: 100B-class targets and tensor parallel across 2×GB10

If you are tempted to serve or train against a **large** Nemotron target (100B-class) split
tensor-parallel across the two Sparks: a ~247 GB BF16 checkpoint does **not** fit TP=2 on
2×GB10 — half of it (~115 GB/rank) plus CUDA context and OS exceeds the per-node unified budget,
and a ZeRO-3 sharded attempt wedges both nodes at full memory. The documented path for a large
target on Spark hardware is the **NVFP4** build on a *single* Spark (with an NVFP4-aware LoRA
path if you fine-tune it), and TP=2 for *inference* only. For the Nano-class target this recipe
targets, none of this applies — it fits and trains on one Spark.

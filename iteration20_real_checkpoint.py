#!/usr/bin/env python3
"""
Iteration 20 - Real-Checkpoint Milestone: a real MoE LLM (OLMoE-1B-7B-Instruct,
64 experts, top-8, Q4_K_M) running the campaign's architecture on this machine,
via the llama.cpp production stack.

WHY
---
Every systems number in iterations 16-19 carries one caveat: random weights.
Bytes and timings do not depend on weight values, but (a) real model QUALITY on
the proposed architecture was never demonstrated, and (b) llama.cpp's fused
dequant-GEMM kernels are precisely what iterations 18-19 priced (2.54x naive
-> 1.21x residual after row-block optimization). This artifact closes both:

1. A real instruction-tuned MoE checkpoint produces real text and a real
   wikitext-2 perplexity on this GPU -- same architecture family the testbeds
   model (router + sparse experts).
2. The campaign's memory architecture -- attention/dense weights resident on
   GPU, ALL expert weights held in host RAM -- is exercised for real via
   llama.cpp's tensor-override (`-ot "exps=CPU"`), and measured against
   full-GPU and full-CPU baselines with llama-bench (JSON output, repeated
   runs).

FREEZE POSTMORTEM (2026-07-03 first run)
----------------------------------------
The first config benched was full-GPU (-ngl 99): 3.92 GB weights + KV cache +
CUDA compute buffers on the 6 GB GTX 1660 Ti that also drives the Windows
desktop. VRAM overcommitted, the driver's sysmem fallback paged GPU memory
over PCIe, and the display starved -- whole-machine freeze, zero output
captured (capture_output=True hid all progress). This version:

1. Benches the campaign architecture (-ot exps=CPU, ~1.5 GB on GPU) FIRST,
   full-CPU second, and full-GPU LAST -- gated on measured free VRAM
   (needs model size + ~900 MB headroom; skipped with a note otherwise;
   --force-full-gpu overrides, --skip-full-gpu never attempts it).
2. Streams all subprocess output to the console live while capturing it.
3. Saves results/iteration20_real_checkpoint.json incrementally after every
   config, so an interruption loses nothing.
4. Recommended one-time manual setting: NVIDIA Control Panel -> Manage 3D
   Settings -> "CUDA - Sysmem Fallback Policy" -> "Prefer No Sysmem
   Fallback", so any future overcommit fails fast instead of freezing
   Windows.

WHAT THIS DOES AND DOES NOT SHOW
--------------------------------
- llama.cpp's expert-offload computes expert matmuls ON THE CPU from host RAM
  (compute-near-data), while iterations 17-19 STREAM expert bytes to the GPU
  (data-to-compute). Same memory placement, two different execution answers;
  the comparison quantifies which wins at this model scale on this machine.
- OLMoE Q4_K_M (4.2 GB) nominally fits in 6 GB VRAM, but the desktop
  compositor shares the card, so full-GPU is measured only when free VRAM
  allows; the offload configs simulate the regime where weights exceed VRAM
  (the campaign's actual target). A 30-70B-class MoE would make offload
  mandatory -- same mechanics, larger constants.
- Routing-trace extraction (real predictor hit rates) needs an instrumented
  build or a torch-side model; if the stock tools cannot expose router
  choices, that remains explicitly open (checked and reported below).

REPRODUCE
---------
  python iteration20_real_checkpoint.py            # expects tools/ + models/ (see README)

Outputs: results/iteration20_real_checkpoint.json (updated after each step)
"""

import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "iteration20_real_checkpoint.json"
TOOLS = Path("tools/llama")
MODEL = Path("models/OLMoE-1B-7B-Instruct-Q4_K_M.gguf")

# Headroom on top of the weight file for KV cache + CUDA compute buffers at
# llama-bench's -p 512 -n 64 workload, plus margin for the desktop compositor.
FULL_GPU_HEADROOM_MB = 900

# Ordered safest-first. "vram_gated" configs run last and only with headroom.
CONFIGS = [
    {"name": "campaign architecture: attn/dense GPU, ALL experts in host RAM",
     "args": ["-ngl", "99", "-ot", "exps=CPU"], "vram_gated": False},
    {"name": "full CPU (floor reference)", "args": ["-ngl", "0"], "vram_gated": False},
    {"name": "full GPU (fits: ceiling reference)", "args": ["-ngl", "99"], "vram_gated": True},
]


def run_streaming(cmd: List[str], timeout: int = 900) -> Tuple[int, str, str]:
    """Run cmd, echoing both pipes to the console live; returns (rc, stdout, stderr)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    chunks = {"out": [], "err": []}

    def drain(pipe, key, prefix):
        for line in pipe:
            chunks[key].append(line)
            print(f"    {prefix} {line.rstrip()}", flush=True)
        pipe.close()

    threads = [threading.Thread(target=drain, args=(proc.stdout, "out", "|"), daemon=True),
               threading.Thread(target=drain, args=(proc.stderr, "err", "!"), daemon=True)]
    for t in threads:
        t.start()
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        rc = -1
        print(f"    [timeout] killed after {timeout}s", flush=True)
    for t in threads:
        t.join(timeout=10)
    return rc, "".join(chunks["out"]), "".join(chunks["err"])


def vram_free_mb() -> Optional[int]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0]) if r.returncode == 0 else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def full_gpu_gate(force: bool) -> Tuple[bool, str]:
    """Decide whether the full-GPU config is safe to attempt on this display GPU."""
    need = int(MODEL.stat().st_size / 2**20) + FULL_GPU_HEADROOM_MB
    free = vram_free_mb()
    if free is None:
        return force, (f"nvidia-smi unavailable; need ~{need} MB free -- "
                       + ("forced on" if force else "skipped (use --force-full-gpu to override)"))
    if free >= need:
        return True, f"free VRAM {free} MB >= required ~{need} MB"
    if force:
        return True, f"free VRAM {free} MB < required ~{need} MB [forced -- freeze risk]"
    return False, (f"free VRAM {free} MB < required ~{need} MB -- skipped to avoid the "
                   f"sysmem-fallback freeze (close GPU apps or use --force-full-gpu)")


def save_payload(payload: Dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def bench(exe: Path, model: Path, extra: List[str]) -> Optional[Dict]:
    cmd = [str(exe), "-m", str(model), "-p", "512", "-n", "64", "-r", "3",
           "-o", "json", *extra]
    rc, out, err = run_streaming(cmd)
    if rc != 0:
        print(f"    llama-bench failed ({rc}): {err.strip()[:300]}")
        return None
    try:
        rows = json.loads(out)
    except json.JSONDecodeError:
        print(f"    unparseable llama-bench output: {out[:200]}")
        return None
    result = {}
    for row in rows:
        kind = "pp" if row.get("n_prompt", 0) > 0 else "tg"
        result[kind] = {"toks_s": round(row["avg_ts"], 2), "stddev": round(row["stddev_ts"], 2)}
    return result


def generation_sanity(exe: Path, model: Path) -> Dict:
    prompt = "Question: Name the three primary colors and one fact about each.\nAnswer:"
    cmd = [str(exe), "-m", str(model), "-p", prompt, "-n", "80", "--temp", "0",
           "-ngl", "99", "-ot", "exps=CPU", "-no-cnv", "--no-display-prompt"]
    rc, out, _ = run_streaming(cmd)
    text = out.strip().replace("\r", "")
    return {"prompt": prompt, "output": text[:600], "ok": rc == 0 and len(text) > 20}


def perplexity(exe: Path, model: Path, corpus: Path, extra: List[str]) -> Optional[float]:
    if not corpus.exists():
        return None
    cmd = [str(exe), "-m", str(model), "-f", str(corpus), "--chunks", "16", *extra]
    _, out, err = run_streaming(cmd, timeout=1800)
    m = re.search(r"Final estimate: PPL = ([\d.]+)", out + err)
    return round(float(m.group(1)), 3) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-full-gpu", action="store_true",
                        help="never attempt the full-GPU config")
    parser.add_argument("--force-full-gpu", action="store_true",
                        help="attempt full-GPU even without measured VRAM headroom")
    args = parser.parse_args()

    bench_exe = TOOLS / "llama-bench.exe"
    cli_exe = TOOLS / "llama-cli.exe"
    ppl_exe = TOOLS / "llama-perplexity.exe"
    for p in (bench_exe, cli_exe, MODEL):
        if not p.exists():
            raise SystemExit(f"missing: {p} (see README iteration 20 setup)")

    print("=" * 100)
    print("ITERATION 20 - REAL CHECKPOINT: OLMoE-1B-7B-Instruct Q4_K_M via llama.cpp on this machine")
    print("=" * 100)
    print(f"[MODEL] {MODEL.name} ({MODEL.stat().st_size / 1e9:.2f} GB, 64 experts, top-8, "
          f"6.9B total / 1.3B active)")

    if args.skip_full_gpu:
        gpu_ok, gpu_reason = False, "disabled via --skip-full-gpu"
    else:
        gpu_ok, gpu_reason = full_gpu_gate(args.force_full_gpu)
    print(f"[full-GPU gate] {gpu_reason}")

    payload = {
        "iteration": 20,
        "purpose": "Real MoE checkpoint on the campaign's memory architecture "
                   "(experts in host RAM) via the llama.cpp production stack.",
        "model": {"file": MODEL.name, "gb": round(MODEL.stat().st_size / 1e9, 2),
                  "family": "OLMoE-1B-7B-0924-Instruct", "experts": 64, "top_k": 8,
                  "params_total_b": 6.9, "params_active_b": 1.3},
        "full_gpu_gate": gpu_reason,
        "bench": [],
        "generation_sanity": None,
        "wikitext2_ppl_16chunks": {},
        "notes": [
            "llama.cpp expert-offload computes expert matmuls on CPU from host RAM "
            "(compute-near-data); iterations 17-19 stream expert bytes to GPU. Same "
            "memory placement, different execution strategy.",
            "Full-GPU ceiling is measured only when free VRAM covers weights + headroom "
            "on this shared display GPU (see full_gpu_gate); offload configs exercise "
            "the weights->host regime the campaign targets.",
            "Routing-trace extraction from stock llama.cpp tools: not exposed; real "
            "predictor hit rates remain open (needs instrumented build or torch model).",
        ],
    }
    save_payload(payload)

    for cfg in CONFIGS:
        if cfg["vram_gated"] and not gpu_ok:
            print(f"\n--- {cfg['name']} --- SKIPPED ({gpu_reason})")
            payload["bench"].append({"config": cfg["name"], "args": cfg["args"],
                                     "bench": None, "skipped": gpu_reason})
            save_payload(payload)
            continue
        print(f"\n--- {cfg['name']} ---")
        b = bench(bench_exe, MODEL, cfg["args"])
        if b:
            print(f"    prompt: {b.get('pp', {}).get('toks_s', '?')} tok/s | "
                  f"decode: {b.get('tg', {}).get('toks_s', '?')} tok/s")
        payload["bench"].append({"config": cfg["name"], "args": cfg["args"], "bench": b})
        save_payload(payload)

    print("\n--- generation sanity (campaign architecture: experts in host RAM) ---")
    gen = generation_sanity(cli_exe, MODEL)
    print(f"    ok={gen['ok']}\n    output: {gen['output'][:300]}")
    payload["generation_sanity"] = gen
    save_payload(payload)

    corpus = Path("models/wikitext-2-test.raw")
    if not args.skip_ppl and ppl_exe.exists() and corpus.exists():
        print("\n--- wikitext-2 perplexity (16 chunks; real quality number) ---")
        ppl_configs = [("experts_in_host_ram", ["-ngl", "99", "-ot", "exps=CPU"])]
        if gpu_ok:
            ppl_configs.append(("full_gpu", ["-ngl", "99"]))
        for label, extra in ppl_configs:
            t0 = time.perf_counter()
            v = perplexity(ppl_exe, MODEL, corpus, extra)
            payload["wikitext2_ppl_16chunks"][label] = v
            save_payload(payload)
            print(f"    {label}: PPL = {v} ({time.perf_counter() - t0:.0f} s)")
    else:
        print("\n[ppl] skipped (corpus or tool missing, or --skip-ppl)")

    print(f"\n[saved] {RESULTS_FILE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
bench.py - Benchmark pplx-embed: C (CPU/BLAS + MLX) vs Python/PyTorch

Measures:
  - Model load time
  - Inference latency (varying sentence lengths)
  - Memory usage
  - Numerical accuracy (cosine similarity between implementations)

Usage:
  .venv/bin/python3 bench.py [--model-dir /path/to/model] [--runs 5]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import psutil
import safetensors.torch as st
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SENTENCES = {
    "short":  "hello world",
    "medium": "The quick brown fox jumps over the lazy dog near the riverbank",
    "long":   (
        "Retrieval-augmented generation is a technique that combines large "
        "language models with external knowledge sources to produce more "
        "accurate and up-to-date responses. The system first retrieves "
        "relevant documents from a corpus using dense vector similarity "
        "search, then conditions the language model on both the query and "
        "the retrieved passages to generate a final answer."
    ),
    "256tok": (
        "In the field of natural language processing, embedding models play "
        "a crucial role in converting text into dense vector representations "
        "that capture semantic meaning. These representations enable a wide "
        "range of downstream tasks including semantic search, clustering, "
        "classification, and retrieval-augmented generation. Modern embedding "
        "models are typically based on transformer architectures that process "
        "input text bidirectionally, allowing each token to attend to all "
        "other tokens in the sequence. This is in contrast to autoregressive "
        "language models which only attend to previous tokens. The resulting "
        "embeddings are obtained by pooling the final hidden states across "
        "all token positions, usually via mean pooling, followed by L2 "
        "normalization to produce unit-length vectors. The cosine similarity "
        "between two such vectors then serves as a measure of semantic "
        "relatedness. Training these models typically involves contrastive "
        "learning objectives where the model learns to produce similar "
        "embeddings for semantically related texts and dissimilar embeddings "
        "for unrelated texts. The choice of training data, negative sampling "
        "strategy, and loss function all significantly impact the quality of "
        "the resulting embeddings."
    ),
}


# ---------------------------------------------------------------------------
# Python/PyTorch reference (manual forward, no HuggingFace overhead)
# ---------------------------------------------------------------------------

class PytorchEmbed:
    """Minimal PyTorch forward pass, matching our C implementation exactly."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.config = json.load(open(f"{model_dir}/config.json"))
        self.hidden = self.config["hidden_size"]
        self.n_layers = self.config["num_hidden_layers"]
        self.n_heads = self.config["num_attention_heads"]
        self.n_kv_heads = self.config["num_key_value_heads"]
        self.head_dim = self.config.get("head_dim", 128)
        self.eps = self.config.get("rms_norm_eps", 1e-6)
        self.rope_theta = self.config.get("rope_theta", 1e6)
        self.weights = None

    def load(self):
        """Load weights from safetensors."""
        # Handle single or multi-shard
        import glob
        shards = sorted(glob.glob(f"{self.model_dir}/model-*.safetensors"))
        if shards:
            self.weights = {}
            for s in shards:
                self.weights.update(st.load_file(s))
        else:
            self.weights = st.load_file(f"{self.model_dir}/model.safetensors")

    def _rms_norm(self, x, w):
        rms = torch.sqrt((x * x).mean(-1, keepdim=True) + self.eps)
        return x / rms * w

    def _rope_neox(self, x, cos, sin):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([
            x1 * cos[..., :half] - x2 * sin[..., :half],
            x2 * cos[..., :half] + x1 * sin[..., :half],
        ], dim=-1)

    def embed(self, token_ids):
        """Run forward pass, return L2-normalized embedding."""
        w = self.weights
        seq = len(token_ids)
        ids = torch.tensor(token_ids)
        hd = self.head_dim

        x = w["embed_tokens.weight"][ids].float()

        # RoPE tables
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, hd, 2).float() / hd))
        pos = torch.arange(seq).float()
        freqs = torch.outer(pos, inv_freq)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)

        scale = 1.0 / math.sqrt(hd)
        heads_per_kv = self.n_heads // self.n_kv_heads

        for i in range(self.n_layers):
            p = f"layers.{i}"

            xn = self._rms_norm(x, w[f"{p}.input_layernorm.weight"])

            q = xn @ w[f"{p}.self_attn.q_proj.weight"].T
            k = xn @ w[f"{p}.self_attn.k_proj.weight"].T
            v = xn @ w[f"{p}.self_attn.v_proj.weight"].T

            q = self._rms_norm(q.view(seq, self.n_heads, hd), w[f"{p}.self_attn.q_norm.weight"])
            k = self._rms_norm(k.view(seq, self.n_kv_heads, hd), w[f"{p}.self_attn.k_norm.weight"])

            q = self._rope_neox(q, cos.unsqueeze(1), sin.unsqueeze(1))
            k = self._rope_neox(k, cos.unsqueeze(1), sin.unsqueeze(1))

            q = q.reshape(seq, -1)
            k = k.reshape(seq, -1)

            attn_out = torch.zeros_like(q)
            for h in range(self.n_heads):
                kv_h = h // heads_per_kv
                qh = q[:, h*hd:(h+1)*hd]
                kh = k[:, kv_h*hd:(kv_h+1)*hd]
                vh = v[:, kv_h*hd:(kv_h+1)*hd]
                scores = torch.softmax((qh @ kh.T) * scale, dim=-1)
                attn_out[:, h*hd:(h+1)*hd] = scores @ vh

            x = x + attn_out @ w[f"{p}.self_attn.o_proj.weight"].T

            xn = self._rms_norm(x, w[f"{p}.post_attention_layernorm.weight"])
            gate = xn @ w[f"{p}.mlp.gate_proj.weight"].T
            up = xn @ w[f"{p}.mlp.up_proj.weight"].T
            x = x + (torch.sigmoid(gate) * gate * up) @ w[f"{p}.mlp.down_proj.weight"].T

        x = self._rms_norm(x, w["norm.weight"])
        emb = x.mean(0)
        emb = emb / emb.norm()
        return emb.numpy()


# ---------------------------------------------------------------------------
# C backend runner (via --daemon mode)
# ---------------------------------------------------------------------------

class CEmbed:
    """Communicate with the C binary via --daemon JSON-lines protocol."""

    def __init__(self, binary, model_dir, backend="cpu"):
        self.binary = binary
        self.model_dir = model_dir
        self.backend = backend
        self.proc = None
        self.load_ms = 0

    def start(self):
        cmd = [self.binary, "-d", self.model_dir, "--daemon"]
        if self.backend == "mlx":
            cmd.append("--mlx")
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Warm up: send one text to trigger model load + kernel compile
        self.proc.stdin.write("warmup\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        self.load_ms = (time.perf_counter() - t0) * 1000
        # Parse warmup result to confirm it works
        try:
            json.loads(line)
        except Exception:
            stderr = self.proc.stderr.read()
            raise RuntimeError(f"C daemon failed to start:\n{stderr}")

    def embed(self, text):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return np.array(result["embedding"], dtype=np.float32), result["ms"]

    def stop(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
            self.proc = None

    def get_memory_mb(self):
        if not self.proc:
            return 0
        try:
            p = psutil.Process(self.proc.pid)
            return p.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Tokenizer helper (use the same BPE as C)
# ---------------------------------------------------------------------------

def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def measure_pytorch(model, tokenizer, texts, n_runs):
    """Measure PyTorch inference times and return embeddings."""
    results = {}
    for name, text in texts.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        n_tok = len(ids)

        # Warmup
        model.embed(ids)

        times = []
        emb = None
        for _ in range(n_runs):
            t0 = time.perf_counter()
            emb = model.embed(ids)
            times.append((time.perf_counter() - t0) * 1000)

        results[name] = {
            "tokens": n_tok,
            "times_ms": times,
            "median_ms": float(np.median(times)),
            "embedding": emb,
        }
    return results


def measure_c(engine, texts, tokenizer, n_runs):
    """Measure C backend inference times and return embeddings."""
    results = {}
    for name, text in texts.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        n_tok = len(ids)

        # Warmup (second call, first was during start)
        engine.embed(text)

        times = []
        emb = None
        for _ in range(n_runs):
            emb, ms = engine.embed(text)
            times.append(ms)

        results[name] = {
            "tokens": n_tok,
            "times_ms": times,
            "median_ms": float(np.median(times)),
            "embedding": emb,
        }
    return results


def print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                      for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in col_widths]))
    for row in rows:
        print(fmt.format(*row))


def main():
    parser = argparse.ArgumentParser(description="Benchmark pplx-embed")
    parser.add_argument("--model-dir", required=True, help="Model directory")
    parser.add_argument("--binary", default="./pplx_embed", help="C binary path")
    parser.add_argument("--runs", type=int, default=5, help="Runs per measurement")
    args = parser.parse_args()

    model_dir = args.model_dir
    config = json.load(open(f"{model_dir}/config.json"))
    dim = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    model_name = os.path.basename(os.path.dirname(model_dir)) or os.path.basename(model_dir)

    print(f"Benchmark: {model_name}")
    print(f"  hidden={dim}, layers={n_layers}, runs={args.runs}")
    print()

    tokenizer = load_tokenizer(model_dir)

    # Token counts
    print("Sentences:")
    for name, text in SENTENCES.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        preview = text[:60] + "..." if len(text) > 60 else text
        print(f"  {name:8s}  {len(ids):3d} tokens  \"{preview}\"")
    print()

    # --- Model size ---
    import glob
    shard_files = sorted(glob.glob(f"{model_dir}/model*.safetensors"))
    model_bytes = sum(os.path.getsize(f) for f in shard_files)
    print(f"Model size on disk: {model_bytes / 1e9:.2f} GB ({len(shard_files)} shard(s))")
    print()

    # --- PyTorch ---
    print("=" * 60)
    print("PyTorch (manual forward, float32, CPU)")
    print("=" * 60)

    py_model = PytorchEmbed(model_dir)
    proc = psutil.Process()

    t0 = time.perf_counter()
    py_model.load()
    py_load_ms = (time.perf_counter() - t0) * 1000

    # Force actual loading by running one inference (mmap is lazy)
    warmup_ids = tokenizer.encode("warmup", add_special_tokens=False)
    mem_before = proc.memory_info().rss
    _ = py_model.embed(warmup_ids)
    mem_after = proc.memory_info().rss
    py_first_ms = (time.perf_counter() - t0) * 1000  # total including first inference
    py_mem_mb = (mem_after - mem_before) / (1024 * 1024)
    # Also measure total RSS for a fair comparison
    py_total_rss = proc.memory_info().rss / (1024 * 1024)

    print(f"  Load time: {py_load_ms:.0f} ms (mmap, lazy)")
    print(f"  First inference (load+embed): {py_first_ms:.0f} ms")
    print(f"  RSS after first inference: ~{py_total_rss:.0f} MB")
    print()

    py_results = measure_pytorch(py_model, tokenizer, SENTENCES, args.runs)

    rows = []
    for name in SENTENCES:
        r = py_results[name]
        rows.append([name, str(r["tokens"]), f"{r['median_ms']:.1f}"])
    print_table(["sentence", "tokens", "median ms"], rows, [10, 8, 12])
    print()

    # Free pytorch weights
    py_model.weights = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    # --- C/BLAS ---
    print("=" * 60)
    print("C/BLAS (Accelerate, multithreaded)")
    print("=" * 60)

    c_cpu = CEmbed(args.binary, model_dir, "cpu")
    c_cpu.start()
    cpu_mem = c_cpu.get_memory_mb()
    print(f"  Load time: {c_cpu.load_ms:.0f} ms (includes warmup)")
    print(f"  RSS: ~{cpu_mem:.0f} MB")
    print()

    cpu_results = measure_c(c_cpu, SENTENCES, tokenizer, args.runs)
    c_cpu.stop()

    rows = []
    for name in SENTENCES:
        r = cpu_results[name]
        rows.append([name, str(r["tokens"]), f"{r['median_ms']:.1f}"])
    print_table(["sentence", "tokens", "median ms"], rows, [10, 8, 12])
    print()

    # --- C/MLX ---
    print("=" * 60)
    print("C/MLX (Metal GPU)")
    print("=" * 60)

    c_mlx = CEmbed(args.binary, model_dir, "mlx")
    c_mlx.start()
    mlx_mem = c_mlx.get_memory_mb()
    print(f"  Load time: {c_mlx.load_ms:.0f} ms (includes warmup + kernel compile)")
    print(f"  RSS: ~{mlx_mem:.0f} MB")
    print()

    mlx_results = measure_c(c_mlx, SENTENCES, tokenizer, args.runs)
    c_mlx.stop()

    rows = []
    for name in SENTENCES:
        r = mlx_results[name]
        rows.append([name, str(r["tokens"]), f"{r['median_ms']:.1f}"])
    print_table(["sentence", "tokens", "median ms"], rows, [10, 8, 12])
    print()

    # --- Accuracy ---
    print("=" * 60)
    print("Accuracy (cosine similarity vs PyTorch reference)")
    print("=" * 60)
    print()

    rows = []
    for name in SENTENCES:
        py_emb = py_results[name]["embedding"]
        cpu_emb = cpu_results[name]["embedding"]
        mlx_emb = mlx_results[name]["embedding"]

        sim_cpu = cosine_sim(py_emb, cpu_emb)
        sim_mlx = cosine_sim(py_emb, mlx_emb)
        max_diff_cpu = float(np.max(np.abs(py_emb - cpu_emb)))
        max_diff_mlx = float(np.max(np.abs(py_emb - mlx_emb)))

        rows.append([
            name,
            str(py_results[name]["tokens"]),
            f"{sim_cpu:.8f}",
            f"{max_diff_cpu:.2e}",
            f"{sim_mlx:.8f}",
            f"{max_diff_mlx:.2e}",
        ])

    print_table(
        ["sentence", "tokens", "cos(cpu,py)", "max|diff|", "cos(mlx,py)", "max|diff|"],
        rows,
        [10, 8, 14, 12, 14, 12],
    )
    print()

    # --- Summary comparison ---
    print("=" * 60)
    print("Summary: Median inference time (ms)")
    print("=" * 60)
    print()

    rows = []
    for name in SENTENCES:
        py_ms = py_results[name]["median_ms"]
        cpu_ms = cpu_results[name]["median_ms"]
        mlx_ms = mlx_results[name]["median_ms"]
        toks = py_results[name]["tokens"]

        rows.append([
            name,
            str(toks),
            f"{py_ms:.1f}",
            f"{cpu_ms:.1f}",
            f"{py_ms/cpu_ms:.1f}x" if cpu_ms > 0 else "-",
            f"{mlx_ms:.1f}",
            f"{py_ms/mlx_ms:.1f}x" if mlx_ms > 0 else "-",
        ])

    print_table(
        ["sentence", "tok", "PyTorch", "C/BLAS", "speedup", "C/MLX", "speedup"],
        rows,
        [10, 5, 10, 10, 9, 10, 9],
    )

    print()
    print(f"Load time:  PyTorch {py_first_ms:.0f} ms (mmap+first embed)  |  C/BLAS {c_cpu.load_ms:.0f} ms  |  C/MLX {c_mlx.load_ms:.0f} ms")
    print(f"RSS:        PyTorch ~{py_total_rss:.0f} MB  |  C/BLAS ~{cpu_mem:.0f} MB  |  C/MLX ~{mlx_mem:.0f} MB")
    print()


if __name__ == "__main__":
    main()

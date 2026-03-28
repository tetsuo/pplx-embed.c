# pplx-embed

Pure C inference for [pplx-embed](https://research.perplexity.ai/articles/pplx-embed-state-of-the-art-embedding-models-for-web-scale-retrieval) embedding models by [Perplexity AI](https://www.perplexity.ai/). Supports safetensors and hardware-accelerated execution (BLAS/mlx-c) for generating float32 vectors.

Based on math kernels from [antirez](https://github.com/antirez)'s [qwen-asr](https://github.com/antirez/qwen-asr).

## Supported models

All [pplx-embed](https://huggingface.co/collections/perplexity-ai/pplx-embed-v1-68126b2e7cc987cd820ebbbd) models are supported.

| Model | Dims | Params | Size |
|---|---|---|---|
| [pplx-embed-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b) | 1024 | 0.6B | ~2.5 GB |
| [pplx-embed-v1-4b](https://huggingface.co/perplexity-ai/pplx-embed-v1-4b) | 2560 | 4B | ~16 GB |
| [pplx-embed-context-v1-0.6B](https://huggingface.co/perplexity-ai/pplx-embed-context-v1-0.6B) | 1024 | 0.6B | ~2.5 GB |
| [pplx-embed-context-v1-4B](https://huggingface.co/perplexity-ai/pplx-embed-context-v1-4B) | 2560 | 4B | ~16 GB |

## Quickstart

```bash
make mlx    # Apple Silicon (fastest)
# or: make blas   # CPU with BLAS (any platform)

./pplx_embed -d /path/to/model "query: what is the capital of France?"
# prints 1024 (or 2560) floats to stdout

./pplx_embed -d /path/to/model \
  "query: what is the capital of France?" \
  "document: Paris is the capital of France." \
  "document: Berlin is the capital of Germany."
# prints cosine similarity matrix
```

### Daemon mode

Start the process once and feed texts through stdin. The model stays loaded in memory.

```bash
# One embedding per line, output as JSON
./pplx_embed -d /path/to/model --daemon --mlx <<'EOF'
query: what is the capital of France?
document: Paris is the capital of France.
document: Berlin is the capital of Germany.
document: Istanbul is the capital of Germany.
EOF
```

Output (one JSON object per line):

```json
{"embedding":[0.0231,-0.0412,...], "dim":1024, "tokens":9, "ms":15.2}
{"embedding":[0.0187,-0.0339,...], "dim":1024, "tokens":9, "ms":14.8}
...
```

## Building

```bash
make blas     # CPU with BLAS (Accelerate on macOS, OpenBLAS on Linux)
make mlx      # Apple Silicon GPU via mlx-c (recommended on Apple Silicon)
make debug    # Debug build with AddressSanitizer
```

For `make mlx`, install MLX first: `brew install mlx mlx-c`

For `make blas` on Linux: `sudo apt install libopenblas-dev`

## Usage

```
./pplx_embed -d <model_dir> [options] [text...]

Options:
  -d <dir>     Model directory (required)
  --mlx        Use Apple MLX GPU backend
  --daemon     Read texts from stdin, write JSON embeddings to stdout
  -t <n>       CPU thread count (default: all cores)
  -e           Print raw embedding vectors (with multiple texts)
  -v           Verbose (-vv for debug)
```

**Single text**: prints the embedding vector as space-separated floats.

**Multiple texts**: prints a cosine similarity matrix.

**Daemon mode** (`--daemon`): reads one text per line from stdin, writes one JSON object per line to stdout. EOF or empty line exits.

## Benchmark results

### pplx-embed-v1-0.6b

1024-dim, 28 layers, 2.38 GB

**Inference latency** (median, 5 runs, Apple M4 Pro 12-core):

| Sentence | Tokens | PyTorch | C/BLAS | Speedup | C/MLX | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| short | 2 | 76.2 ms | 53.9 ms | 1.4x | 15.2 ms | 5.0x |
| medium | 13 | 75.7 ms | 50.7 ms | 1.5x | 15.4 ms | 4.9x |
| long | 63 | 97.4 ms | 58.9 ms | 1.7x | 20.0 ms | 4.9x |
| 256tok | 199 | 193.5 ms | 178.2 ms | 1.1x | 53.9 ms | 3.6x |

**Load/memory**:

| | PyTorch | C/BLAS | C/MLX |
| :--- | :--- | :--- | :--- |
| **Load time** | 186 ms | 195 ms | 363 ms |
| **RSS** | ~2163 MB | ~1737 MB | ~4603 MB |

### pplx-embed-v1-4b

2560-dim, 36 layers, 16.09 GB

| Sentence | Tokens | PyTorch | C/BLAS | Speedup | C/MLX | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| short | 2 | 462 ms | 402 ms | 1.1x | 97 ms | 4.8x |
| medium | 13 | 307 ms | 234 ms | 1.3x | 101 ms | 3.0x |
| long | 63 | 411 ms | 350 ms | 1.2x | 121 ms | 3.4x |
| 256tok | 199 | 875 ms | 919 ms | 1.0x | 376 ms | 2.3x |

| | PyTorch | C/BLAS | C/MLX |
| :--- | :--- | :--- | :--- |
| **Load time** | 12094 ms | 1202 ms | 6134 ms |
| **RSS** | ~14253 MB | ~13902 MB | ~15483 MB |

**Summary**:

* C/MLX is up to **5x** faster than PyTorch on Apple M4 Pro, and C/BLAS is up to **1.7x** faster.
* C/BLAS uses ~20% less memory than PyTorch (1737 vs 2163 MB for 0.6B) because it mmap's weights directly instead of copying them into Python objects.
* MLX RSS is higher (~4.6 GB for 0.6B) because Metal duplicates weight data into GPU-accessible memory. This is the cost of GPU speed.
* Load times: C/BLAS loads fastest (mmap, no conversion). MLX has a one-time Metal kernel compilation cost. PyTorch's 12-second load for 4B is because the first inference faults in the entire mmap'd file.
* Accuracy is identical (cosine sim = 1.000000, max diff < 2e-6) between PyTorch and C implementations.


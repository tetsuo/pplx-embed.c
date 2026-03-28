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


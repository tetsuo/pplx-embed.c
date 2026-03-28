/*
 * pplx_embed_mlx.c - MLX GPU backend for pplx-embed inference
 *
 * Uses mlx-c (Apple's pure C API) for GPU-accelerated transformer inference.
 * The full forward pass runs on Metal: token embed > 28 layers > mean pool > L2 norm.
 *
 * Key operations used:
 *   mlx_take          - embedding lookup
 *   mlx_matmul        - linear projections (no bias)
 *   mlx_fast_rms_norm - RMSNorm (fused Metal kernel)
 *   mlx_fast_rope     - NeoX-style rotary embeddings (fused Metal kernel)
 *   mlx_fast_scaled_dot_product_attention - GQA attention (fused Metal kernel)
 *   sigmoid * gate, multiply - SwiGLU activation
 */

#include "pplx_embed_mlx.h"
#include "pplx_embed.h"           /* config constants */
#include "qwen_asr_safetensors.h" /* safetensors reader */

#include <mlx/c/mlx.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Per-layer weight arrays (MLX)
 * ======================================================================== */

typedef struct {
    mlx_array wq;               /* [q_dim,   hidden] */
    mlx_array wk;               /* [kv_dim,  hidden] */
    mlx_array wv;               /* [kv_dim,  hidden] */
    mlx_array wo;               /* [hidden,  q_dim]  */
    mlx_array q_norm;           /* [head_dim] */
    mlx_array k_norm;           /* [head_dim] */
    mlx_array input_norm;       /* [hidden] */
    mlx_array post_attn_norm;   /* [hidden] */
    mlx_array gate_proj;        /* [intermediate, hidden] */
    mlx_array up_proj;          /* [intermediate, hidden] */
    mlx_array down_proj;        /* [hidden, intermediate] */
} mlx_layer_t;

struct pplx_mlx_ctx {
    mlx_array embed_tokens;     /* [vocab, hidden] */
    mlx_layer_t layers[PPLX_NUM_LAYERS];
    mlx_array norm;             /* [hidden] */
    mlx_stream stream;

    /* Safetensors kept alive for mmap */
    multi_safetensors_t *ms;
};

/* ========================================================================
 * Helpers
 * ======================================================================== */

/* Default GPU stream */
static mlx_stream gpu_stream(pplx_mlx_ctx_t *ctx) { return ctx->stream; }

/*
 * Load an F32 tensor from safetensors into an mlx_array.
 * The data is copied into MLX (which will move it to GPU as needed).
 */
static mlx_array load_tensor(multi_safetensors_t *ms, const char *name,
                              const int *shape, int ndim)
{
    safetensors_file_t *sf = NULL;
    const safetensor_t *t  = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "mlx: tensor not found: %s\n", name);
        return (mlx_array){0};
    }
    if (t->dtype != DTYPE_F32) {
        fprintf(stderr, "mlx: expected F32 for %s\n", name);
        return (mlx_array){0};
    }
    const void *data = safetensors_data(sf, t);
    return mlx_array_new_data(data, shape, ndim, MLX_FLOAT32);
}

/* Quick check: is arr valid? */
static int arr_ok(mlx_array a) { return a.ctx != NULL; }

/* ========================================================================
 * Load
 * ======================================================================== */

pplx_mlx_ctx_t *pplx_mlx_load(const char *model_dir)
{
    pplx_mlx_ctx_t *ctx = calloc(1, sizeof(pplx_mlx_ctx_t));
    if (!ctx) return NULL;

    ctx->stream = mlx_default_gpu_stream_new();

    /* Open safetensors */
    ctx->ms = multi_safetensors_open(model_dir);
    if (!ctx->ms) {
        fprintf(stderr, "mlx: failed to open safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }

    multi_safetensors_t *ms = ctx->ms;

    /* Embedding table: [vocab_size, hidden] */
    int emb_shape[] = {PPLX_VOCAB_SIZE, PPLX_HIDDEN_SIZE};
    ctx->embed_tokens = load_tensor(ms, "embed_tokens.weight", emb_shape, 2);
    if (!arr_ok(ctx->embed_tokens)) goto fail;

    /* Per-layer weights */
    char name[256];
    for (int i = 0; i < PPLX_NUM_LAYERS; i++) {
        mlx_layer_t *l = &ctx->layers[i];
        int hidden = PPLX_HIDDEN_SIZE;
        int q_dim  = PPLX_Q_DIM;
        int kv_dim = PPLX_KV_DIM;
        int inter  = PPLX_INTERMEDIATE_SIZE;
        int hdim   = PPLX_HEAD_DIM;

#define LOAD(field, fmt, ...) do {                                \
    snprintf(name, sizeof(name), fmt, i);                         \
    int sh[] = {__VA_ARGS__};                                     \
    int nd = sizeof(sh) / sizeof(sh[0]);                          \
    l->field = load_tensor(ms, name, sh, nd);                     \
    if (!arr_ok(l->field)) goto fail;                             \
} while (0)

        LOAD(wq,             "layers.%d.self_attn.q_proj.weight", q_dim, hidden);
        LOAD(wk,             "layers.%d.self_attn.k_proj.weight", kv_dim, hidden);
        LOAD(wv,             "layers.%d.self_attn.v_proj.weight", kv_dim, hidden);
        LOAD(wo,             "layers.%d.self_attn.o_proj.weight", hidden, q_dim);
        LOAD(q_norm,         "layers.%d.self_attn.q_norm.weight", hdim);
        LOAD(k_norm,         "layers.%d.self_attn.k_norm.weight", hdim);
        LOAD(input_norm,     "layers.%d.input_layernorm.weight", hidden);
        LOAD(post_attn_norm, "layers.%d.post_attention_layernorm.weight", hidden);
        LOAD(gate_proj,      "layers.%d.mlp.gate_proj.weight", inter, hidden);
        LOAD(up_proj,        "layers.%d.mlp.up_proj.weight", inter, hidden);
        LOAD(down_proj,      "layers.%d.mlp.down_proj.weight", hidden, inter);

#undef LOAD

        if (pplx_verbose >= 2)
            fprintf(stderr, "  mlx layer %d loaded\n", i);
    }

    /* Final norm */
    int norm_shape[] = {PPLX_HIDDEN_SIZE};
    ctx->norm = load_tensor(ms, "norm.weight", norm_shape, 1);
    if (!arr_ok(ctx->norm)) goto fail;

    if (pplx_verbose >= 1)
        fprintf(stderr, "mlx: all weights loaded\n");

    return ctx;

fail:
    pplx_mlx_free(ctx);
    return NULL;
}

/* ========================================================================
 * Free
 * ======================================================================== */

static void free_layer(mlx_layer_t *l)
{
    mlx_array_free(l->wq);
    mlx_array_free(l->wk);
    mlx_array_free(l->wv);
    mlx_array_free(l->wo);
    mlx_array_free(l->q_norm);
    mlx_array_free(l->k_norm);
    mlx_array_free(l->input_norm);
    mlx_array_free(l->post_attn_norm);
    mlx_array_free(l->gate_proj);
    mlx_array_free(l->up_proj);
    mlx_array_free(l->down_proj);
}

void pplx_mlx_free(pplx_mlx_ctx_t *ctx)
{
    if (!ctx) return;
    mlx_array_free(ctx->embed_tokens);
    for (int i = 0; i < PPLX_NUM_LAYERS; i++)
        free_layer(&ctx->layers[i]);
    mlx_array_free(ctx->norm);
    mlx_stream_free(ctx->stream);
    if (ctx->ms) multi_safetensors_close(ctx->ms);
    free(ctx);
}

/* ========================================================================
 * Forward pass  (all on GPU stream, lazy-evaluated)
 *
 * Notation:  S = gpu_stream(ctx)
 *
 * Linear without bias:  y = x @ W^T
 *   Since mlx_matmul(x, W^T) requires transposing W, and our weights
 *   are stored as [out, in], we do:  y = x @ W^T  via
 *     Wt = transpose(W)     [in, out]
 *     y  = matmul(x, Wt)    [seq, out]
 *
 *   Alternatively: y = matmul(x, transpose(W))
 *   All MLX ops are lazy (graph-built) so transpose is free.
 * ======================================================================== */

/* Convenience: transpose a 2D matrix */
static mlx_array t2d(mlx_array a, mlx_stream s)
{
    mlx_array res = mlx_array_new();
    mlx_transpose(&res, a, s);
    return res;
}

/* y = x @ W^T  (linear, no bias) */
static mlx_array linear(mlx_array x, mlx_array W, mlx_stream s)
{
    mlx_array Wt  = t2d(W, s);
    mlx_array res = mlx_array_new();
    mlx_matmul(&res, x, Wt, s);
    mlx_array_free(Wt);
    return res;
}

float *pplx_mlx_embed(pplx_mlx_ctx_t *ctx, const int *token_ids, int n_tokens)
{
    if (!ctx || !token_ids || n_tokens <= 0) return NULL;

    mlx_stream S = gpu_stream(ctx);
    int hidden = PPLX_HIDDEN_SIZE;
    int n_heads    = PPLX_NUM_HEADS;
    int n_kv_heads = PPLX_NUM_KV_HEADS;
    int head_dim   = PPLX_HEAD_DIM;
    int q_dim      = PPLX_Q_DIM;
    int seq        = n_tokens;
    float scale    = 1.0f / sqrtf((float)head_dim);

    (void)n_heads; (void)n_kv_heads; /* used only in shape literals */

    /* ------------------------------------------------------------------
     * 1. Token embedding lookup:  x = embed_tokens[token_ids]
     * ------------------------------------------------------------------ */
    int ids_shape[] = {seq};
    mlx_array ids = mlx_array_new_data(token_ids, ids_shape, 1, MLX_INT32);
    mlx_array x   = mlx_array_new();
    mlx_take_axis(&x, ctx->embed_tokens, ids, /*axis=*/0, S);
    mlx_array_free(ids);
    /* x: [seq, hidden] */

    /* ------------------------------------------------------------------
     * 2. Transformer layers (28x)
     * ------------------------------------------------------------------ */
    for (int layer = 0; layer < PPLX_NUM_LAYERS; layer++) {
        mlx_layer_t *l = &ctx->layers[layer];

        /* ---- 2a. Input RMSNorm ---- */
        mlx_array xn = mlx_array_new();
        mlx_fast_rms_norm(&xn, x, l->input_norm, 1e-6f, S);

        /* ---- 2b. QKV projections (no bias) ----
         *   q: [seq, q_dim]   k: [seq, kv_dim]   v: [seq, kv_dim]
         */
        mlx_array q_flat = linear(xn, l->wq, S);
        mlx_array k_flat = linear(xn, l->wk, S);
        mlx_array v_flat = linear(xn, l->wv, S);
        mlx_array_free(xn);

        /* ---- 2c. Reshape to [seq, n_heads, head_dim] for per-head ops ---- */
        int q_shape[] = {seq, n_heads,    head_dim};
        int k_shape[] = {seq, n_kv_heads, head_dim};

        mlx_array q = mlx_array_new();
        mlx_array k = mlx_array_new();
        mlx_array v = mlx_array_new();
        mlx_reshape(&q, q_flat, q_shape, 3, S);
        mlx_reshape(&k, k_flat, k_shape, 3, S);
        mlx_reshape(&v, v_flat, k_shape, 3, S);
        mlx_array_free(q_flat);
        mlx_array_free(k_flat);
        mlx_array_free(v_flat);

        /* ---- 2d. Per-head Q/K RMSNorm ----
         *   q: [seq, n_heads, head_dim] ; norm over last dim (head_dim)
         */
        mlx_array qn = mlx_array_new();
        mlx_array kn = mlx_array_new();
        mlx_fast_rms_norm(&qn, q, l->q_norm, 1e-6f, S);
        mlx_fast_rms_norm(&kn, k, l->k_norm, 1e-6f, S);
        mlx_array_free(q);
        mlx_array_free(k);

        /* ---- 2e. Transpose to [1, n_heads, seq, head_dim] for RoPE + SDPA ----
         *   MLX rope expects (..., T, D) where T is second-to-last.
         *   [seq, n_heads, head_dim]
         *     - expand_dims(0)        [1, seq, n_heads, head_dim]
         *     - transpose [0,2,1,3]   [1, n_heads, seq, head_dim]
         */
        int perm[] = {0, 2, 1, 3};

        mlx_array qe = mlx_array_new();
        mlx_array ke = mlx_array_new();
        mlx_array ve = mlx_array_new();
        mlx_expand_dims(&qe, qn, 0, S);
        mlx_expand_dims(&ke, kn, 0, S);
        mlx_expand_dims(&ve, v,  0, S);
        mlx_array_free(qn);
        mlx_array_free(kn);

        mlx_array qt = mlx_array_new();
        mlx_array kt = mlx_array_new();
        mlx_array vt = mlx_array_new();
        mlx_transpose_axes(&qt, qe, perm, 4, S);
        mlx_transpose_axes(&kt, ke, perm, 4, S);
        mlx_transpose_axes(&vt, ve, perm, 4, S);
        mlx_array_free(qe);
        mlx_array_free(ke);
        mlx_array_free(ve);
        mlx_array_free(v);

        /* ---- 2f. RoPE (NeoX-style: traditional=false) ----
         *   qt: [1, n_heads,    seq, head_dim]
         *   kt: [1, n_kv_heads, seq, head_dim]
         *   MLX rope: T = second-to-last dim = seq
         */
        mlx_optional_float base = {.value = 1000000.0f, .has_value = true};
        mlx_array null_freqs = (mlx_array){0};

        mlx_array qr = mlx_array_new();
        mlx_array kr = mlx_array_new();
        mlx_fast_rope(&qr, qt, head_dim, /*traditional=*/false,
                      base, /*scale=*/1.0f, /*offset=*/0, null_freqs, S);
        mlx_fast_rope(&kr, kt, head_dim, /*traditional=*/false,
                      base, /*scale=*/1.0f, /*offset=*/0, null_freqs, S);
        mlx_array_free(qt);
        mlx_array_free(kt);

        /* ---- 2g. Bidirectional GQA Attention ----
         *   qr: [1, n_heads,    seq, head_dim]
         *   kr: [1, n_kv_heads, seq, head_dim]
         *   vt: [1, n_kv_heads, seq, head_dim]
         *   mask_mode="" > no causal mask > full bidirectional
         *   GQA: SDPA broadcasts KV heads to match Q heads automatically.
         */
        mlx_array attn = mlx_array_new();
        mlx_array null_mask = (mlx_array){0};
        mlx_array null_sinks = (mlx_array){0};
        mlx_fast_scaled_dot_product_attention(
            &attn, qr, kr, vt, scale,
            /*mask_mode=*/"",
            null_mask, null_sinks, S);
        mlx_array_free(qr);
        mlx_array_free(kr);
        mlx_array_free(vt);
        /* attn: [1, n_heads, seq, head_dim] */

        /* ---- 2h. Transpose back > [1, seq, n_heads, head_dim] > squeeze > reshape [seq, q_dim] ---- */
        mlx_array attn_t = mlx_array_new();
        mlx_transpose_axes(&attn_t, attn, perm, 4, S);  /* [1, seq, n_heads, head_dim] */
        mlx_array_free(attn);

        mlx_array attn_sq = mlx_array_new();
        mlx_squeeze_axis(&attn_sq, attn_t, 0, S);  /* [seq, n_heads, head_dim] */
        mlx_array_free(attn_t);

        int attn_flat_shape[] = {seq, q_dim};
        mlx_array attn_flat = mlx_array_new();
        mlx_reshape(&attn_flat, attn_sq, attn_flat_shape, 2, S);
        mlx_array_free(attn_sq);

        /* ---- 2i. Output projection + residual ---- */
        mlx_array proj = linear(attn_flat, l->wo, S);
        mlx_array_free(attn_flat);

        mlx_array x2 = mlx_array_new();
        mlx_add(&x2, x, proj, S);
        mlx_array_free(x);
        mlx_array_free(proj);
        x = x2;

        /* ---- 2j. Post-attention RMSNorm ---- */
        mlx_array xn2 = mlx_array_new();
        mlx_fast_rms_norm(&xn2, x, l->post_attn_norm, 1e-6f, S);

        /* ---- 2k. SwiGLU MLP ----
         *   gate = x_norm @ gate_proj^T    [seq, inter]
         *   up   = x_norm @ up_proj^T      [seq, inter]
         *   mid  = silu(gate) * up  =  (gate * sigmoid(gate)) * up
         *   out  = mid @ down_proj^T        [seq, hidden]
         */
        mlx_array gate = linear(xn2, l->gate_proj, S);
        mlx_array up   = linear(xn2, l->up_proj, S);
        mlx_array_free(xn2);

        /* SiLU(gate) = gate * sigmoid(gate) */
        mlx_array gate_sig = mlx_array_new();
        mlx_sigmoid(&gate_sig, gate, S);
        mlx_array silu_gate = mlx_array_new();
        mlx_multiply(&silu_gate, gate, gate_sig, S);
        mlx_array_free(gate);
        mlx_array_free(gate_sig);

        /* mid = silu_gate * up */
        mlx_array mid = mlx_array_new();
        mlx_multiply(&mid, silu_gate, up, S);
        mlx_array_free(silu_gate);
        mlx_array_free(up);

        /* down projection */
        mlx_array ffn = linear(mid, l->down_proj, S);
        mlx_array_free(mid);

        /* Residual */
        mlx_array x3 = mlx_array_new();
        mlx_add(&x3, x, ffn, S);
        mlx_array_free(x);
        mlx_array_free(ffn);
        x = x3;
    }

    /* ------------------------------------------------------------------
     * 3. Final RMSNorm
     * ------------------------------------------------------------------ */
    mlx_array x_normed = mlx_array_new();
    mlx_fast_rms_norm(&x_normed, x, ctx->norm, 1e-6f, S);
    mlx_array_free(x);

    /* ------------------------------------------------------------------
     * 4. Mean pooling: mean(x, axis=0)  [hidden]
     * ------------------------------------------------------------------ */
    mlx_array emb = mlx_array_new();
    mlx_mean_axis(&emb, x_normed, /*axis=*/0, /*keepdims=*/false, S);
    mlx_array_free(x_normed);

    /* ------------------------------------------------------------------
     * 5. L2 normalization: emb / ||emb||_2
     * ------------------------------------------------------------------ */
    int norm_axes[] = {0};
    mlx_array norm_val = mlx_array_new();
    mlx_linalg_norm_l2(&norm_val, emb, norm_axes, 1, /*keepdims=*/true, S);

    mlx_array emb_normed = mlx_array_new();
    mlx_divide(&emb_normed, emb, norm_val, S);
    mlx_array_free(emb);
    mlx_array_free(norm_val);

    /* ------------------------------------------------------------------
     * 6. Evaluate and copy result to CPU
     * ------------------------------------------------------------------ */
    mlx_array_eval(emb_normed);

    const float *data = mlx_array_data_float32(emb_normed);
    if (!data) {
        fprintf(stderr, "mlx: failed to get embedding data\n");
        mlx_array_free(emb_normed);
        return NULL;
    }

    float *out = (float *)malloc(hidden * sizeof(float));
    if (out) memcpy(out, data, hidden * sizeof(float));

    mlx_array_free(emb_normed);
    return out;
}

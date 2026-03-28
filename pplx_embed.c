/*
 * pplx_embed.c - Pure C inference for perplexity-ai/pplx-embed-v1-0.6b
 *
 * Transformer forward pass:
 *   embed -> 28x[RMSNorm -> QKV (no bias) -> Q/K RMSNorm -> RoPE
 *              -> Bidirectional GQA attention -> O proj -> residual
 *              -> RMSNorm -> SwiGLU MLP -> residual]
 *   -> final RMSNorm -> mean pool -> L2 normalize
 *
 * Key implementation choices:
 *   - All weights are F32 (mmap'd direct, no copy)
 *   - Bidirectional attention: reuse qwen_causal_attention with
 *     q_offset = (seq_len - 1), which removes the causal mask
 *     (every query sees all seq_len keys)
 *   - GQA ratio 2:1 (16 Q-heads, 8 KV-heads) handled by causal_attention
 *   - RoPE cache grown lazily, positions [0 .. seq_len-1]
 */

#include "pplx_embed.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Global verbose flags
 * qwen_verbose is declared extern in qwen_asr_kernels.h and qwen_asr_tokenizer.c;
 * we own its definition here since we don't link qwen_asr.c.
 * ======================================================================== */

int pplx_verbose = 0;
int qwen_verbose = 0;  /* required by qwen_asr_kernels.c and qwen_asr_tokenizer.c */

/* ========================================================================
 * Internal helpers: weight loading
 * ======================================================================== */

/*
 * Return a direct pointer into the mmap'd safetensors region for an F32 tensor.
 * No allocation, no copy; caller must NOT free the returned pointer.
 */
static const float *load_f32_direct(multi_safetensors_t *ms, const char *name)
{
    safetensors_file_t *sf = NULL;
    const safetensor_t *t  = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "pplx_embed: tensor not found: %s\n", name);
        return NULL;
    }
    if (t->dtype != DTYPE_F32) {
        fprintf(stderr, "pplx_embed: expected F32 for %s, got dtype=%d\n",
                name, (int)t->dtype);
        return NULL;
    }
    return (const float *)safetensors_data(sf, t);
}

/* ========================================================================
 * Internal helpers: working buffers
 * ======================================================================== */

static int ensure_buffers(pplx_ctx_t *ctx, int seq)
{
    if (seq <= ctx->buf_seq_cap) return 0;

    int new_cap = ctx->buf_seq_cap > 0 ? ctx->buf_seq_cap : 64;
    while (new_cap < seq) new_cap *= 2;

    int hidden = ctx->config.hidden_size;
    int q_dim  = ctx->config.q_dim;
    int kv_dim = ctx->config.kv_dim;
    int inter  = ctx->config.intermediate_size;

#define PPLX_REALLOC(ptr, n) do {                                   \
    void *_tmp = realloc((ptr), (size_t)(n) * sizeof(float));       \
    if (!_tmp) return -1;                                            \
    (ptr) = (float *)_tmp;                                           \
} while (0)

    PPLX_REALLOC(ctx->x,        new_cap * hidden);
    PPLX_REALLOC(ctx->x_norm,   new_cap * hidden);
    PPLX_REALLOC(ctx->q,        new_cap * q_dim);
    PPLX_REALLOC(ctx->k,        new_cap * kv_dim);
    PPLX_REALLOC(ctx->v,        new_cap * kv_dim);
    PPLX_REALLOC(ctx->attn_out, new_cap * q_dim);
    PPLX_REALLOC(ctx->proj_out, new_cap * hidden);
    PPLX_REALLOC(ctx->ffn_gate, new_cap * inter);
    PPLX_REALLOC(ctx->ffn_up,   new_cap * inter);
    PPLX_REALLOC(ctx->ffn_out,  new_cap * hidden);

#undef PPLX_REALLOC

    ctx->buf_seq_cap = new_cap;
    return 0;
}

/* ========================================================================
 * Internal helpers: RoPE cache
 *
 * Stores cos/sin for positions [0 .. rope_cache_cap-1].
 * Layout: [pos, head_dim] - values are duplicated across the two halves
 * so that qwen_apply_rope_neox can load contiguous cos[d] and cos[half+d]
 * from a single row: cos_row[d] == cos_row[half+d].
 * ======================================================================== */

static int ensure_rope_cache(pplx_ctx_t *ctx, int n_pos)
{
    if (n_pos <= ctx->rope_cache_cap) return 0;

    int new_cap = ctx->rope_cache_cap > 0 ? ctx->rope_cache_cap : 512;
    while (new_cap < n_pos) new_cap *= 2;

    int   head_dim = ctx->config.head_dim;
    int   half     = head_dim / 2;
    float theta    = ctx->config.rope_theta;

    size_t n = (size_t)new_cap * head_dim;
    float *new_cos = (float *)realloc(ctx->rope_cos, n * sizeof(float));
    float *new_sin = (float *)realloc(ctx->rope_sin, n * sizeof(float));
    if (!new_cos || !new_sin) {
        /* If realloc failed, partial state is still valid up to old cap */
        if (new_cos) ctx->rope_cos = new_cos;
        if (new_sin) ctx->rope_sin = new_sin;
        return -1;
    }
    ctx->rope_cos = new_cos;
    ctx->rope_sin = new_sin;

    /* Compute only the newly needed positions */
    for (int pos = ctx->rope_cache_cap; pos < new_cap; pos++) {
        float fp = (float)pos;
        float *c  = ctx->rope_cos + (size_t)pos * head_dim;
        float *s  = ctx->rope_sin + (size_t)pos * head_dim;
        for (int d = 0; d < half; d++) {
            /* inv_freq[d] = 1 / (theta ^ (2d / head_dim)) */
            float inv_freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
            float angle    = fp * inv_freq;
            float cv = cosf(angle);
            float sv = sinf(angle);
            /* Duplicate across both halves for the NeoX rotation kernel */
            c[d]        = cv;
            c[half + d] = cv;
            s[d]        = sv;
            s[half + d] = sv;
        }
    }

    ctx->rope_cache_cap = new_cap;
    return 0;
}

/* ========================================================================
 * Load
 * ======================================================================== */

pplx_ctx_t *pplx_load(const char *model_dir)
{
    pplx_ctx_t *ctx = (pplx_ctx_t *)calloc(1, sizeof(pplx_ctx_t));
    if (!ctx) return NULL;

    /* ---- Hardcoded config (mirrors config.json) ---- */
    pplx_config_t *cfg     = &ctx->config;
    cfg->hidden_size       = PPLX_HIDDEN_SIZE;
    cfg->n_layers          = PPLX_NUM_LAYERS;
    cfg->n_heads           = PPLX_NUM_HEADS;
    cfg->n_kv_heads        = PPLX_NUM_KV_HEADS;
    cfg->head_dim          = PPLX_HEAD_DIM;
    cfg->q_dim             = PPLX_Q_DIM;
    cfg->kv_dim            = PPLX_KV_DIM;
    cfg->intermediate_size = PPLX_INTERMEDIATE_SIZE;
    cfg->vocab_size        = PPLX_VOCAB_SIZE;
    cfg->rms_norm_eps      = 1e-6f;
    cfg->rope_theta        = 1000000.0f;

    /* ---- Open safetensors ---- */
    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "pplx_load: failed to open safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }
    ctx->safetensors = ms;

    if (pplx_verbose >= 1)
        fprintf(stderr, "pplx_load: loading weights from %s\n", model_dir);

    /* ---- Token embeddings ---- */
    pplx_weights_t *w    = &ctx->weights;
    w->embed_tokens = load_f32_direct(ms, "embed_tokens.weight");
    if (!w->embed_tokens) goto fail;

    /* ---- Per-layer weights ---- */
    char name[256];
    for (int i = 0; i < cfg->n_layers; i++) {
        pplx_layer_t *l = &w->layers[i];

#define LOAD(field, fmt) do {                               \
    snprintf(name, sizeof(name), fmt, i);                   \
    l->field = load_f32_direct(ms, name);                   \
    if (!l->field) goto fail;                               \
} while (0)

        LOAD(wq,             "layers.%d.self_attn.q_proj.weight");
        LOAD(wk,             "layers.%d.self_attn.k_proj.weight");
        LOAD(wv,             "layers.%d.self_attn.v_proj.weight");
        LOAD(wo,             "layers.%d.self_attn.o_proj.weight");
        LOAD(q_norm,         "layers.%d.self_attn.q_norm.weight");
        LOAD(k_norm,         "layers.%d.self_attn.k_norm.weight");
        LOAD(input_norm,     "layers.%d.input_layernorm.weight");
        LOAD(post_attn_norm, "layers.%d.post_attention_layernorm.weight");
        LOAD(gate_proj,      "layers.%d.mlp.gate_proj.weight");
        LOAD(up_proj,        "layers.%d.mlp.up_proj.weight");
        LOAD(down_proj,      "layers.%d.mlp.down_proj.weight");

#undef LOAD

        if (pplx_verbose >= 2)
            fprintf(stderr, "  layer %d loaded\n", i);
    }

    /* ---- Final RMSNorm ---- */
    w->norm = load_f32_direct(ms, "norm.weight");
    if (!w->norm) goto fail;

    if (pplx_verbose >= 1)
        fprintf(stderr, "pplx_load: all weights loaded\n");

    return ctx;

fail:
    pplx_free(ctx);
    return NULL;
}

/* ========================================================================
 * Free
 * ======================================================================== */

void pplx_free(pplx_ctx_t *ctx)
{
    if (!ctx) return;
    if (ctx->safetensors)
        multi_safetensors_close((multi_safetensors_t *)ctx->safetensors);

    free(ctx->x);
    free(ctx->x_norm);
    free(ctx->q);
    free(ctx->k);
    free(ctx->v);
    free(ctx->attn_out);
    free(ctx->proj_out);
    free(ctx->ffn_gate);
    free(ctx->ffn_up);
    free(ctx->ffn_out);
    free(ctx->rope_cos);
    free(ctx->rope_sin);
    free(ctx);
}

/* ========================================================================
 * Forward pass
 * ======================================================================== */

float *pplx_embed(pplx_ctx_t *ctx, const int *token_ids, int n_tokens)
{
    if (!ctx || !token_ids || n_tokens <= 0) return NULL;

    const pplx_config_t  *cfg = &ctx->config;
    const pplx_weights_t *w   = &ctx->weights;

    int hidden = cfg->hidden_size;
    int n_heads   = cfg->n_heads;
    int n_kv_heads = cfg->n_kv_heads;
    int head_dim  = cfg->head_dim;
    int q_dim     = cfg->q_dim;
    int kv_dim    = cfg->kv_dim;
    int inter     = cfg->intermediate_size;
    float eps     = cfg->rms_norm_eps;
    int seq       = n_tokens;

    /* Allocate/grow working buffers */
    if (ensure_buffers(ctx, seq) != 0) {
        fprintf(stderr, "pplx_embed: buffer allocation failed\n");
        return NULL;
    }

    /* Pre-compute RoPE tables for positions [0, seq-1] */
    if (ensure_rope_cache(ctx, seq) != 0) {
        fprintf(stderr, "pplx_embed: RoPE cache allocation failed\n");
        return NULL;
    }
    const float *rope_cos = ctx->rope_cos; /* [seq_cap, head_dim] */
    const float *rope_sin = ctx->rope_sin;

    float *x        = ctx->x;
    float *x_norm   = ctx->x_norm;
    float *q        = ctx->q;
    float *k        = ctx->k;
    float *v        = ctx->v;
    float *attn_out = ctx->attn_out;
    float *proj_out = ctx->proj_out;
    float *ffn_gate = ctx->ffn_gate;
    float *ffn_up   = ctx->ffn_up;
    float *ffn_out  = ctx->ffn_out;

    /* ------------------------------------------------------------------
     * Step 1: Token embedding lookup
     *   x[i, :] = embed_tokens[token_ids[i], :]
     * ------------------------------------------------------------------ */
    for (int i = 0; i < seq; i++) {
        int id = token_ids[i];
        if (id < 0 || id >= cfg->vocab_size) {
            fprintf(stderr, "pplx_embed: invalid token id %d at position %d\n", id, i);
            return NULL;
        }
        memcpy(x + i * hidden,
               w->embed_tokens + (size_t)id * hidden,
               hidden * sizeof(float));
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    /* ------------------------------------------------------------------
     * Steps 2..29: 28 transformer layers
     * ------------------------------------------------------------------ */
    for (int layer = 0; layer < cfg->n_layers; layer++) {
        const pplx_layer_t *l = &w->layers[layer];

        /* ---- 2a. Input RMSNorm ---- */
        qwen_rms_norm(x_norm, x, l->input_norm, seq, hidden, eps);

        /* ---- 2b. QKV projections (no bias) ----
         *   q: [seq, q_dim]   = x_norm @ wq^T   (wq is [q_dim, hidden])
         *   k: [seq, kv_dim]  = x_norm @ wk^T
         *   v: [seq, kv_dim]  = x_norm @ wv^T
         */
        qwen_linear_nobias(q, x_norm, l->wq, seq, hidden, q_dim);
        qwen_linear_nobias(k, x_norm, l->wk, seq, hidden, kv_dim);
        qwen_linear_nobias(v, x_norm, l->wv, seq, hidden, kv_dim);

        /* ---- 2c. Per-head Q/K RMSNorm (Qwen3 feature) ----
         *   Normalize each head's [head_dim] segment independently.
         */
        qwen_rms_norm_per_head(q, l->q_norm, seq, n_heads,    head_dim, eps);
        qwen_rms_norm_per_head(k, l->k_norm, seq, n_kv_heads, head_dim, eps);

        /* ---- 2d. Apply NeoX-style RoPE to Q and K ----
         *   rope_cos/sin: [seq_cap, head_dim], positions 0..seq-1 cached
         */
        qwen_apply_rope_neox(q, rope_cos, rope_sin, seq, n_heads,    head_dim);
        qwen_apply_rope_neox(k, rope_cos, rope_sin, seq, n_kv_heads, head_dim);

        /* ---- 2e. Bidirectional GQA attention ----
         *
         * We reuse qwen_causal_attention with q_offset = (seq - 1).
         *
         * Inside the kernel for query position i:
         *   global_pos = q_offset + i = (seq-1) + i
         *   k_end      = min(global_pos + 1, seq_k)
         *              = min(seq + i,        seq)   = seq   for all i >= 0
         *
         * So every query attends to ALL seq key positions - full attention,
         * no causal mask, while still benefiting from the GQA + threading
         * in qwen_causal_attention.
         */
        qwen_causal_attention(attn_out, q, k, v,
                              seq,   /* seq_q */
                              seq,   /* seq_k */
                              n_heads, n_kv_heads, head_dim,
                              scale,
                              seq - 1); /* q_offset - removes causal mask */

        /* ---- 2f. Output projection + residual ----
         *   proj_out = attn_out @ wo^T   (wo is [hidden, q_dim])
         *   x += proj_out
         */
        qwen_linear_nobias(proj_out, attn_out, l->wo, seq, q_dim, hidden);
        qwen_add_inplace(x, proj_out, seq * hidden);

        /* ---- 2g. Post-attention RMSNorm ---- */
        qwen_rms_norm(x_norm, x, l->post_attn_norm, seq, hidden, eps);

        /* ---- 2h. SwiGLU MLP ----
         *   gate = gate_proj(x_norm)   [seq, intermediate]
         *   up   = up_proj(x_norm)     [seq, intermediate]
         *   mid  = SiLU(gate) * up     [seq, intermediate]
         *   out  = down_proj(mid)      [seq, hidden]
         */
        qwen_linear_nobias(ffn_gate, x_norm, l->gate_proj, seq, hidden, inter);
        qwen_linear_nobias(ffn_up,   x_norm, l->up_proj,   seq, hidden, inter);

        qwen_silu(ffn_gate, seq * inter);                  /* SiLU(gate) in-place */
        qwen_mul_inplace(ffn_gate, ffn_up, seq * inter);   /* gate *= up */

        qwen_linear_nobias(ffn_out, ffn_gate, l->down_proj, seq, inter, hidden);
        qwen_add_inplace(x, ffn_out, seq * hidden);
    }

    /* ------------------------------------------------------------------
     * Step 3: Final RMSNorm (in-place on x)
     * ------------------------------------------------------------------ */
    qwen_rms_norm(x, x, w->norm, seq, hidden, eps);

    /* ------------------------------------------------------------------
     * Step 4: Mean pooling across all token positions
     *   embedding[d] = mean over i in [0, seq) of x[i, d]
     * ------------------------------------------------------------------ */
    float *embedding = (float *)calloc(hidden, sizeof(float));
    if (!embedding) return NULL;

    for (int i = 0; i < seq; i++) {
        const float *row = x + i * hidden;
        for (int d = 0; d < hidden; d++) {
            embedding[d] += row[d];
        }
    }
    float inv_seq = 1.0f / (float)seq;
    for (int d = 0; d < hidden; d++) embedding[d] *= inv_seq;

    /* ------------------------------------------------------------------
     * Step 5: L2 normalization
     *   embedding /= ||embedding||_2
     * ------------------------------------------------------------------ */
    float norm_sq = 0.0f;
    for (int d = 0; d < hidden; d++) norm_sq += embedding[d] * embedding[d];
    float norm_val = sqrtf(norm_sq);
    if (norm_val > 1e-8f) {
        float inv = 1.0f / norm_val;
        for (int d = 0; d < hidden; d++) embedding[d] *= inv;
    }

    return embedding;
}

/* ========================================================================
 * Cosine similarity (dot product of two L2-normalized vectors)
 * ======================================================================== */

float pplx_cosine_similarity(const float *a, const float *b, int dim)
{
    float dot = 0.0f;
    for (int i = 0; i < dim; i++) dot += a[i] * b[i];
    return dot;
}

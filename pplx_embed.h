/*
 * pplx_embed.h - Pure C inference for perplexity-ai/pplx-embed-v1-0.6b
 *
 * Architecture: Qwen3-family transformer with BIDIRECTIONAL (non-causal)
 * attention. Based on antirez's qwen-asr kernels. Produces dense float32
 * embedding vectors via mean-pooling + L2-normalization.
 *
 * Model spec:
 *   hidden_size                 = 1024
 *   num_hidden_layers           = 28
 *   num_attention_heads         = 16   (query heads)
 *   num_key_value_heads         = 8    (GQA: 2 query heads share 1 KV head)
 *   head_dim                    = 128
 *   intermediate_size           = 3072
 *   vocab_size                  = 151936
 *   rms_norm_eps                = 1e-6
 *   rope_theta                  = 1000000.0 (1e6)
 *   weights dtype               = F32 (all weights float32)
 *   tie_word_embeddings         = true
 *   use_bidirectional_attention = true
 */

#ifndef PPLX_EMBED_H
#define PPLX_EMBED_H

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * Constants
 * ======================================================================== */

#define PPLX_HIDDEN_SIZE        1024
#define PPLX_NUM_LAYERS         28
#define PPLX_NUM_HEADS          16
#define PPLX_NUM_KV_HEADS       8
#define PPLX_HEAD_DIM           128
#define PPLX_Q_DIM              (PPLX_NUM_HEADS * PPLX_HEAD_DIM)    /* 2048 */
#define PPLX_KV_DIM             (PPLX_NUM_KV_HEADS * PPLX_HEAD_DIM) /* 1024 */
#define PPLX_INTERMEDIATE_SIZE  3072
#define PPLX_VOCAB_SIZE         151936

/* ========================================================================
 * Model Configuration
 * ======================================================================== */

typedef struct {
    int hidden_size;        /* 1024 */
    int n_layers;           /* 28   */
    int n_heads;            /* 16   */
    int n_kv_heads;         /* 8    */
    int head_dim;           /* 128  */
    int q_dim;              /* n_heads    * head_dim = 2048 */
    int kv_dim;             /* n_kv_heads * head_dim = 1024 */
    int intermediate_size;  /* 3072 */
    int vocab_size;         /* 151936 */
    float rms_norm_eps;     /* 1e-6 */
    float rope_theta;       /* 1e6  */
} pplx_config_t;

/* ========================================================================
 * Per-Layer Weights  (all F32, direct mmap pointers - read-only)
 *
 * Tensor shapes (row-major):
 *   wq           [q_dim,   hidden]       = [2048, 1024]
 *   wk           [kv_dim,  hidden]       = [1024, 1024]
 *   wv           [kv_dim,  hidden]       = [1024, 1024]
 *   wo           [hidden,  q_dim]        = [1024, 2048]
 *   q_norm       [head_dim]              = [128]
 *   k_norm       [head_dim]              = [128]
 *   input_norm   [hidden]               = [1024]
 *   post_attn_norm [hidden]             = [1024]
 *   gate_proj    [intermediate, hidden] = [3072, 1024]
 *   up_proj      [intermediate, hidden] = [3072, 1024]
 *   down_proj    [hidden, intermediate] = [1024, 3072]
 * ======================================================================== */

typedef struct {
    /* Self-attention projections (no bias) */
    const float *wq;            /* [q_dim,   hidden] */
    const float *wk;            /* [kv_dim,  hidden] */
    const float *wv;            /* [kv_dim,  hidden] */
    const float *wo;            /* [hidden,  q_dim]  */

    /* Per-head Q/K RMSNorm weights (Qwen3 feature) */
    const float *q_norm;        /* [head_dim] */
    const float *k_norm;        /* [head_dim] */

    /* Layer norms (no bias - RMSNorm only) */
    const float *input_norm;    /* [hidden] pre-attention */
    const float *post_attn_norm;/* [hidden] pre-MLP */

    /* SwiGLU MLP (no bias) */
    const float *gate_proj;     /* [intermediate, hidden] */
    const float *up_proj;       /* [intermediate, hidden] */
    const float *down_proj;     /* [hidden, intermediate] */
} pplx_layer_t;

/* ========================================================================
 * Full Model Weights
 * ======================================================================== */

typedef struct {
    /* Token embedding table (tied with output head, but we don't need lm_head) */
    const float *embed_tokens;          /* [vocab_size, hidden] */

    /* Transformer layers */
    pplx_layer_t layers[PPLX_NUM_LAYERS];

    /* Final RMSNorm */
    const float *norm;                  /* [hidden] */
} pplx_weights_t;

/* ========================================================================
 * Context (model + working buffers)
 * ======================================================================== */

typedef struct {
    pplx_config_t  config;
    pplx_weights_t weights;

    /* Open safetensors handle (keeps mmap alive) */
    void *safetensors;          /* multi_safetensors_t* */

    /* Working buffers, grown lazily to buf_seq_cap */
    int    buf_seq_cap;
    float *x;                   /* [seq, hidden]       main activations */
    float *x_norm;              /* [seq, hidden]       after RMSNorm */
    float *q;                   /* [seq, q_dim]        query vectors */
    float *k;                   /* [seq, kv_dim]       key vectors */
    float *v;                   /* [seq, kv_dim]       value vectors */
    float *attn_out;            /* [seq, q_dim]        attention output */
    float *proj_out;            /* [seq, hidden]       after o_proj */
    float *ffn_gate;            /* [seq, intermediate] gate branch */
    float *ffn_up;              /* [seq, intermediate] up branch */
    float *ffn_out;             /* [seq, hidden]       after down_proj */

    /* RoPE cosine/sine cache [n_pos, head_dim], grown lazily */
    float *rope_cos;
    float *rope_sin;
    int    rope_cache_cap;      /* number of cached positions */
} pplx_ctx_t;

/* ========================================================================
 * API
 * ======================================================================== */

/*
 * Load the model from a directory containing:
 *   model.safetensors - weight tensors (F32)
 *   vocab.json        - BPE vocabulary
 *   merges.txt        - BPE merge rules
 *
 * Returns NULL on error.
 */
pplx_ctx_t *pplx_load(const char *model_dir);

/* Free all resources */
void pplx_free(pplx_ctx_t *ctx);

/*
 * Compute the embedding for a sequence of token IDs.
 *
 * Performs:
 *   1. Token embedding lookup
 *   2. 28 × bidirectional transformer layers (GQA, RoPE, SwiGLU)
 *   3. Final RMSNorm
 *   4. Mean pooling across all token positions
 *   5. L2 normalization
 *
 * Returns a malloc'd float[hidden_size] (caller must free).
 * Returns NULL on allocation error or empty input.
 */
float *pplx_embed(pplx_ctx_t *ctx, const int *token_ids, int n_tokens);

/*
 * Cosine similarity between two L2-normalized embedding vectors.
 * (This is just the dot product since both are unit vectors.)
 */
float pplx_cosine_similarity(const float *a, const float *b, int dim);

/* Verbose level: 0=quiet, 1=info, 2=debug */
extern int pplx_verbose;
extern int qwen_verbose;  /* same flag, used internally by kernel/tokenizer */

#endif /* PPLX_EMBED_H */

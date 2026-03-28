/*
 * pplx_embed_mlx.h - MLX GPU backend for pplx-embed inference
 *
 * Uses Apple's mlx-c to run the transformer on Metal GPU.
 * All operations run on the GPU default stream; the result is copied
 * back to CPU as a float* embedding.
 */

#ifndef PPLX_EMBED_MLX_H
#define PPLX_EMBED_MLX_H

#include <stddef.h>
#include <stdint.h>

/* Opaque MLX model context */
typedef struct pplx_mlx_ctx pplx_mlx_ctx_t;

/*
 * Load model into MLX arrays from safetensors in model_dir.
 * Returns NULL on error.
 */
pplx_mlx_ctx_t *pplx_mlx_load(const char *model_dir);

/*
 * Free all MLX resources.
 */
void pplx_mlx_free(pplx_mlx_ctx_t *ctx);

/*
 * Compute embedding for token_ids[0..n_tokens-1].
 * Returns malloc'd float[1024] (caller frees). NULL on error.
 */
float *pplx_mlx_embed(pplx_mlx_ctx_t *ctx, const int *token_ids, int n_tokens);

#endif /* PPLX_EMBED_MLX_H */

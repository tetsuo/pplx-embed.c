/*
 * main.c - pplx_embed command-line tool
 *
 * Usage:
 *   pplx_embed -d <model_dir> [options] <text> [<text2> ...]
 *   pplx_embed -d <model_dir> [options]              (read lines from stdin)
 *
 * Options:
 *   -d <dir>   Model directory (required)
 *   -t <n>     Number of threads (default: all CPU cores)
 *   -e         Print raw embedding vector(s) instead of similarity matrix
 *   -v         Verbose output (use -vv for debug)
 *
 * Output:
 *   1 text  : prints the 1024-dim embedding as space-separated floats
 *   2+ texts: prints cosine-similarity matrix between every pair,
 *             then optionally the individual embeddings if -e is given
 *
 * Example:
 *   ./pplx_embed -d /path/to/model "hello world" "hi there"
 *   ./pplx_embed -d /path/to/model -e "query: what is AI?"
 */

#include "pplx_embed.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_tokenizer.h"

#ifdef USE_MLX
#include "pplx_embed_mlx.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ========================================================================
 * Helpers
 * ======================================================================== */

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void print_embedding(const float *emb, int dim)
{
    for (int i = 0; i < dim; i++) {
        if (i > 0) putchar(' ');
        printf("%.8f", (double)emb[i]);
    }
    putchar('\n');
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s -d <model_dir> [options] [text...]\n"
        "\n"
        "Options:\n"
        "  -d <dir>   Model directory (model.safetensors + vocab.json + merges.txt)\n"
        "  -t <n>     Number of threads (default: all CPU cores, CPU backend)\n"
        "  --mlx      Use Apple MLX GPU backend (Apple Silicon only)\n"
        "  -e         Always print raw embedding vector(s)\n"
        "  -v         Verbose (repeat for more: -vv)\n"
        "  -h         Show this help\n"
        "\n"
        "With 1 text:  prints the 1024-dim embedding vector\n"
        "With 2+ texts: prints cosine similarity between each pair\n"
        "               (add -e to also print the individual embeddings)\n"
        "\n"
        "If no text arguments are given, reads one text per line from stdin.\n"
        "\n"
        "Example:\n"
        "  %s -d ./model \"query: what is AI?\" \"document: AI is artificial intelligence\"\n"
        "  %s -d ./model --mlx \"query: what is AI?\"\n",
        prog, prog, prog);
}

/* ========================================================================
 * Process one text: tokenize -> embed, returns malloc'd float[dim]
 * ======================================================================== */

static float *process_text(pplx_ctx_t        *ctx,
#ifdef USE_MLX
                            pplx_mlx_ctx_t    *mlx_ctx,
#endif
                            qwen_tokenizer_t  *tok,
                            const char        *text,
                            int                print_tokens)
{
    /* Tokenize */
    int  n_tokens  = 0;
    int *token_ids = qwen_tokenizer_encode(tok, text, &n_tokens);
    if (!token_ids || n_tokens == 0) {
        fprintf(stderr, "pplx_embed: tokenization failed for: %s\n", text);
        free(token_ids);
        return NULL;
    }

    if (print_tokens || pplx_verbose >= 1) {
        fprintf(stderr, "tokens (%d): ", n_tokens);
        for (int i = 0; i < n_tokens && i < 20; i++)
            fprintf(stderr, "%d ", token_ids[i]);
        if (n_tokens > 20) fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }

    /* Embed */
    double t0   = now_ms();
    float *emb  = NULL;
#ifdef USE_MLX
    if (mlx_ctx)
        emb = pplx_mlx_embed(mlx_ctx, token_ids, n_tokens);
    else
#endif
        emb = pplx_embed(ctx, token_ids, n_tokens);
    double dt   = now_ms() - t0;
    free(token_ids);

    if (!emb) {
        fprintf(stderr, "pplx_embed: forward pass failed\n");
        return NULL;
    }

    if (pplx_verbose >= 1)
        fprintf(stderr, "embed: %d tokens in %.1f ms\n", n_tokens, dt);

    return emb;
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(int argc, char *argv[])
{
    const char *model_dir   = NULL;
    int         n_threads   = 0;   /* 0 = auto */
    int         print_embs  = 0;
    int         verbose     = 0;
    int         use_mlx     = 0;

    /* ---- Parse flags ---- */
    int arg_start = 1;
    while (arg_start < argc && argv[arg_start][0] == '-') {
        const char *flag = argv[arg_start];
        if (strcmp(flag, "-d") == 0) {
            if (arg_start + 1 >= argc) {
                fprintf(stderr, "pplx_embed: -d requires an argument\n");
                return 1;
            }
            model_dir = argv[++arg_start];
        } else if (strcmp(flag, "-t") == 0) {
            if (arg_start + 1 >= argc) {
                fprintf(stderr, "pplx_embed: -t requires an argument\n");
                return 1;
            }
            n_threads = atoi(argv[++arg_start]);
        } else if (strcmp(flag, "-e") == 0) {
            print_embs = 1;
        } else if (strcmp(flag, "--mlx") == 0) {
#ifdef USE_MLX
            use_mlx = 1;
#else
            fprintf(stderr, "pplx_embed: --mlx not available (build with: make mlx)\n");
            return 1;
#endif
        } else if (strcmp(flag, "-v") == 0) {
            verbose++;
        } else if (strcmp(flag, "-vv") == 0) {
            verbose = 2;
        } else if (strcmp(flag, "-h") == 0 || strcmp(flag, "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            /* Stop flag parsing on first non-flag argument */
            break;
        }
        arg_start++;
    }

    if (!model_dir || model_dir[0] == '\0') {
        fprintf(stderr, "pplx_embed: model directory required (-d <dir>)\n"
                        "  Hint: if using MODEL=.. ./pplx_embed -d \"$MODEL\",\n"
                        "  make sure MODEL is exported or use: MODEL=.. && ./pplx_embed -d \"$MODEL\"\n");
        print_usage(argv[0]);
        return 1;
    }

    /* ---- Set verbosity ---- */
    pplx_verbose  = verbose;
    qwen_verbose  = verbose;   /* kernels share the same flag name */

    /* ---- Set thread count (CPU backend only) ---- */
    if (!use_mlx) {
        if (n_threads <= 0)
            n_threads = qwen_get_num_cpus();
        qwen_set_threads(n_threads);
        if (verbose >= 1)
            fprintf(stderr, "Using %d CPU thread(s)\n", n_threads);
    } else {
        if (verbose >= 1)
            fprintf(stderr, "Using MLX GPU backend\n");
    }

    /* ---- Load tokenizer ---- */
    char vocab_path[1024];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", model_dir);

    double t0_load = now_ms();
    qwen_tokenizer_t *tok = qwen_tokenizer_load(vocab_path);
    if (!tok) {
        fprintf(stderr, "pplx_embed: failed to load tokenizer from %s\n", vocab_path);
        return 1;
    }
    if (verbose >= 1)
        fprintf(stderr, "Tokenizer loaded in %.0f ms\n", now_ms() - t0_load);

    /* ---- Load model ---- */
    pplx_ctx_t *ctx = NULL;
#ifdef USE_MLX
    pplx_mlx_ctx_t *mlx_ctx = NULL;
#endif

    double t0_model = now_ms();

#ifdef USE_MLX
    if (use_mlx) {
        mlx_ctx = pplx_mlx_load(model_dir);
        if (!mlx_ctx) {
            fprintf(stderr, "pplx_embed: failed to load MLX model from %s\n", model_dir);
            qwen_tokenizer_free(tok);
            return 1;
        }
    } else
#endif
    {
        ctx = pplx_load(model_dir);
        if (!ctx) {
            fprintf(stderr, "pplx_embed: failed to load model from %s\n", model_dir);
            qwen_tokenizer_free(tok);
            return 1;
        }
    }
    if (verbose >= 1)
        fprintf(stderr, "Model loaded in %.0f ms%s\n", now_ms() - t0_model,
                use_mlx ? " (MLX)" : "");

    int dim = PPLX_HIDDEN_SIZE;

    /* ---- Gather input texts ---- */
    /* texts[] and embeddings[] are grown dynamically */
    int     n_texts   = 0;
    int     texts_cap = 0;
    char  **texts     = NULL;
    float **embeddings = NULL;

    /* Helper to append a text */
#define APPEND_TEXT(s) do {                                             \
    if (n_texts == texts_cap) {                                         \
        texts_cap = texts_cap ? texts_cap * 2 : 8;                     \
        texts      = realloc(texts,      texts_cap * sizeof(char *));   \
        embeddings = realloc(embeddings, texts_cap * sizeof(float *));  \
        if (!texts || !embeddings) {                                    \
            fprintf(stderr, "pplx_embed: out of memory\n"); return 1;  \
        }                                                               \
    }                                                                   \
    texts[n_texts]      = strdup(s);                                    \
    embeddings[n_texts] = NULL;                                         \
    n_texts++;                                                          \
} while (0)

    if (arg_start < argc) {
        /* Text arguments on command line */
        for (int i = arg_start; i < argc; i++)
            APPEND_TEXT(argv[i]);
    } else {
        /* Read lines from stdin */
        char line[65536];
        while (fgets(line, sizeof(line), stdin)) {
            /* Strip trailing newline */
            size_t l = strlen(line);
            while (l > 0 && (line[l-1] == '\n' || line[l-1] == '\r'))
                line[--l] = '\0';
            if (l == 0) continue;
            APPEND_TEXT(line);
        }
    }

    if (n_texts == 0) {
        fprintf(stderr, "pplx_embed: no input texts\n");
        pplx_free(ctx);
        qwen_tokenizer_free(tok);
        return 1;
    }

    /* ---- Embed all texts ---- */
    for (int i = 0; i < n_texts; i++) {
        if (verbose >= 1)
            fprintf(stderr, "[%d/%d] \"%s\"\n", i+1, n_texts, texts[i]);

        embeddings[i] = process_text(ctx,
#ifdef USE_MLX
                                     mlx_ctx,
#endif
                                     tok, texts[i],
                                     /*print_tokens=*/ verbose >= 2);
        if (!embeddings[i]) {
            /* Error already printed */
            goto cleanup;
        }
    }

    /* ---- Output ---- */
    if (n_texts == 1) {
        /* Single text: always print the embedding */
        print_embedding(embeddings[0], dim);
    } else {
        /* Multiple texts: print cosine similarity matrix */
        printf("Cosine similarity matrix (%d texts):\n", n_texts);

        /* Header */
        printf("%-4s", "");
        for (int j = 0; j < n_texts; j++) printf("  [%d]  ", j);
        printf("\n");

        for (int i = 0; i < n_texts; i++) {
            printf("[%d] ", i);
            for (int j = 0; j < n_texts; j++) {
                float sim = pplx_cosine_similarity(embeddings[i], embeddings[j], dim);
                printf("  %.4f", (double)sim);
            }
            printf("  \"%s\"\n", texts[i]);
        }

        /* Print raw embeddings if requested */
        if (print_embs) {
            printf("\nEmbeddings:\n");
            for (int i = 0; i < n_texts; i++) {
                printf("[%d] ", i);
                print_embedding(embeddings[i], dim);
            }
        }
    }

cleanup:
    for (int i = 0; i < n_texts; i++) {
        free(texts[i]);
        free(embeddings[i]);
    }
    free(texts);
    free(embeddings);
    if (ctx) pplx_free(ctx);
#ifdef USE_MLX
    if (mlx_ctx) pplx_mlx_free(mlx_ctx);
#endif
    qwen_tokenizer_free(tok);
    return 0;
}

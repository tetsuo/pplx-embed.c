/* main.c - pplx_embed command-line tool */

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

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s -d <model_dir> [options] [text...]\n"
        "\n"
        "Options:\n"
        "  -d <dir>     Model directory (required)\n"
        "  --mlx        Use Apple MLX GPU backend\n"
        "  --daemon     Read lines from stdin, write JSON embeddings to stdout\n"
        "  -t <n>       CPU threads (default: all cores)\n"
        "  -e           Print raw embeddings (with multiple texts)\n"
        "  -v           Verbose (-vv for debug)\n"
        "  -h           Show this help\n"
        "\n"
        "Modes:\n"
        "  1  text arg     Print embedding as space-separated floats\n"
        "  2+ text args    Print cosine similarity matrix\n"
        "  no args         Batch: read all stdin lines, then similarity matrix\n"
        "  --daemon        Streaming: read stdin lines, write JSON per line\n"
        "\n"
        "Examples:\n"
        "  %s -d ./model \"query: what is AI?\"\n"
        "  %s -d ./model --mlx --daemon < texts.txt\n",
        prog, prog, prog);
}

/* ========================================================================
 * Embed one text: tokenize > forward > return float[dim]
 * ======================================================================== */

typedef struct {
    pplx_ctx_t       *ctx;
#ifdef USE_MLX
    pplx_mlx_ctx_t   *mlx_ctx;
#endif
    qwen_tokenizer_t *tok;
    int               dim;
} engine_t;

typedef struct {
    float *emb;       /* malloc'd float[dim], caller frees */
    int    n_tokens;
    double ms;
} embed_result_t;

static embed_result_t embed_text(engine_t *e, const char *text)
{
    embed_result_t r = {0};

    int n = 0;
    int *ids = qwen_tokenizer_encode(e->tok, text, &n);
    if (!ids || n == 0) {
        fprintf(stderr, "tokenization failed: %s\n", text);
        free(ids);
        return r;
    }

    if (pplx_verbose >= 1) {
        fprintf(stderr, "tokens (%d): ", n);
        for (int i = 0; i < n && i < 20; i++) fprintf(stderr, "%d ", ids[i]);
        if (n > 20) fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }

    double t0 = now_ms();
#ifdef USE_MLX
    if (e->mlx_ctx)
        r.emb = pplx_mlx_embed(e->mlx_ctx, ids, n);
    else
#endif
        r.emb = pplx_embed(e->ctx, ids, n);
    r.ms = now_ms() - t0;
    r.n_tokens = n;
    free(ids);

    if (!r.emb) fprintf(stderr, "forward pass failed\n");
    if (pplx_verbose >= 1)
        fprintf(stderr, "embed: %d tokens in %.1f ms\n", r.n_tokens, r.ms);

    return r;
}

/* ========================================================================
 * Output helpers
 * ======================================================================== */

static void print_embedding_raw(const float *emb, int dim)
{
    for (int i = 0; i < dim; i++) {
        if (i > 0) putchar(' ');
        printf("%.8f", (double)emb[i]);
    }
    putchar('\n');
}

static void print_embedding_json(const float *emb, int dim, int n_tokens, double ms)
{
    printf("{\"embedding\":[");
    for (int i = 0; i < dim; i++) {
        if (i > 0) putchar(',');
        printf("%.8f", (double)emb[i]);
    }
    printf("],\"dim\":%d,\"tokens\":%d,\"ms\":%.1f}\n", dim, n_tokens, ms);
    fflush(stdout);
}

/* ========================================================================
 * Daemon mode: read lines from stdin, write JSON to stdout
 * ======================================================================== */

static int run_daemon(engine_t *e)
{
    char line[65536];

    if (pplx_verbose >= 1)
        fprintf(stderr, "daemon: ready, reading from stdin\n");

    while (fgets(line, sizeof(line), stdin)) {
        /* Strip newline */
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        if (pplx_verbose >= 1)
            fprintf(stderr, "daemon: \"%.*s%s\"\n",
                    (int)(len > 60 ? 60 : len), line, len > 60 ? "..." : "");

        embed_result_t r = embed_text(e, line);
        if (r.emb) {
            print_embedding_json(r.emb, e->dim, r.n_tokens, r.ms);
            free(r.emb);
        } else {
            /* Output an error line so the consumer stays in sync */
            printf("{\"error\":\"embedding failed\"}\n");
            fflush(stdout);
        }
    }

    if (pplx_verbose >= 1)
        fprintf(stderr, "daemon: stdin EOF\n");

    return 0;
}

/* ========================================================================
 * Batch mode: embed args or stdin lines, then print similarity or vectors
 * ======================================================================== */

static int run_batch(engine_t *e, int argc, char **argv, int arg_start, int print_embs)
{
    int     n_texts = 0, cap = 0;
    char  **texts = NULL;
    float **embs  = NULL;

#define APPEND(s) do {                                          \
    if (n_texts == cap) {                                        \
        cap = cap ? cap * 2 : 8;                                 \
        texts = realloc(texts, cap * sizeof(char *));            \
        embs  = realloc(embs,  cap * sizeof(float *));           \
        if (!texts || !embs) { fprintf(stderr, "OOM\n"); return 1; } \
    }                                                            \
    texts[n_texts] = strdup(s);                                  \
    embs[n_texts]  = NULL;                                       \
    n_texts++;                                                   \
} while (0)

    if (arg_start < argc) {
        for (int i = arg_start; i < argc; i++) APPEND(argv[i]);
    } else {
        char line[65536];
        while (fgets(line, sizeof(line), stdin)) {
            size_t l = strlen(line);
            while (l > 0 && (line[l-1] == '\n' || line[l-1] == '\r'))
                line[--l] = '\0';
            if (l == 0) continue;
            APPEND(line);
        }
    }
#undef APPEND

    if (n_texts == 0) {
        fprintf(stderr, "no input texts\n");
        free(texts); free(embs);
        return 1;
    }

    int dim = e->dim;

    for (int i = 0; i < n_texts; i++) {
        if (pplx_verbose >= 1)
            fprintf(stderr, "[%d/%d] \"%s\"\n", i+1, n_texts, texts[i]);
        embed_result_t r = embed_text(e, texts[i]);
        embs[i] = r.emb;
        if (!embs[i]) goto done;
    }

    if (n_texts == 1) {
        print_embedding_raw(embs[0], dim);
    } else {
        printf("Cosine similarity matrix (%d texts):\n", n_texts);
        printf("%-4s", "");
        for (int j = 0; j < n_texts; j++) printf("  [%d]  ", j);
        printf("\n");
        for (int i = 0; i < n_texts; i++) {
            printf("[%d] ", i);
            for (int j = 0; j < n_texts; j++)
                printf("  %.4f", (double)pplx_cosine_similarity(embs[i], embs[j], dim));
            printf("  \"%s\"\n", texts[i]);
        }
        if (print_embs) {
            printf("\nEmbeddings:\n");
            for (int i = 0; i < n_texts; i++) {
                printf("[%d] ", i);
                print_embedding_raw(embs[i], dim);
            }
        }
    }

done:
    for (int i = 0; i < n_texts; i++) { free(texts[i]); free(embs[i]); }
    free(texts); free(embs);
    return 0;
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(int argc, char *argv[])
{
    const char *model_dir = NULL;
    int n_threads  = 0;
    int print_embs = 0;
    int verbose    = 0;
    int use_mlx    = 0;
    int daemon     = 0;

    int arg_start = 1;
    while (arg_start < argc && argv[arg_start][0] == '-') {
        const char *f = argv[arg_start];
        if      (!strcmp(f, "-d"))      { model_dir = argv[++arg_start]; }
        else if (!strcmp(f, "-t"))      { n_threads = atoi(argv[++arg_start]); }
        else if (!strcmp(f, "-e"))      { print_embs = 1; }
        else if (!strcmp(f, "-v"))      { verbose++; }
        else if (!strcmp(f, "-vv"))     { verbose = 2; }
        else if (!strcmp(f, "--daemon")){ daemon = 1; }
        else if (!strcmp(f, "--mlx"))   {
#ifdef USE_MLX
            use_mlx = 1;
#else
            fprintf(stderr, "--mlx not available (build with: make mlx)\n"); return 1;
#endif
        }
        else if (!strcmp(f, "-h") || !strcmp(f, "--help")) { print_usage(argv[0]); return 0; }
        else break;
        arg_start++;
    }

    if (!model_dir || !model_dir[0]) {
        fprintf(stderr, "model directory required (-d <dir>)\n");
        print_usage(argv[0]);
        return 1;
    }

    pplx_verbose = verbose;
    qwen_verbose = verbose;

    /* Threads (CPU backend) */
    if (!use_mlx) {
        if (n_threads <= 0) n_threads = qwen_get_num_cpus();
        qwen_set_threads(n_threads);
        if (verbose >= 1) fprintf(stderr, "Using %d CPU thread(s)\n", n_threads);
    } else {
        if (verbose >= 1) fprintf(stderr, "Using MLX GPU backend\n");
    }

    /* Tokenizer */
    char vocab_path[1024];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", model_dir);
    double t0 = now_ms();
    qwen_tokenizer_t *tok = qwen_tokenizer_load(vocab_path);
    if (!tok) { fprintf(stderr, "failed to load tokenizer: %s\n", vocab_path); return 1; }
    if (verbose >= 1) fprintf(stderr, "Tokenizer: %.0f ms\n", now_ms() - t0);

    /* Model */
    engine_t e = {0};
    e.tok = tok;
    t0 = now_ms();

#ifdef USE_MLX
    if (use_mlx) {
        e.mlx_ctx = pplx_mlx_load(model_dir);
        if (!e.mlx_ctx) { fprintf(stderr, "failed to load model\n"); return 1; }
        e.dim = pplx_mlx_config(e.mlx_ctx)->hidden_size;
    } else
#endif
    {
        e.ctx = pplx_load(model_dir);
        if (!e.ctx) { fprintf(stderr, "failed to load model\n"); return 1; }
        e.dim = e.ctx->config.hidden_size;
    }
    if (verbose >= 1)
        fprintf(stderr, "Model: %d-dim, %.0f ms%s\n",
                e.dim, now_ms() - t0, use_mlx ? " (MLX)" : "");

    /* Run */
    int rc;
    if (daemon)
        rc = run_daemon(&e);
    else
        rc = run_batch(&e, argc, argv, arg_start, print_embs);

    /* Cleanup */
    if (e.ctx) pplx_free(e.ctx);
#ifdef USE_MLX
    if (e.mlx_ctx) pplx_mlx_free(e.mlx_ctx);
#endif
    qwen_tokenizer_free(tok);
    return rc;
}

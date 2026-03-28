# pplx_embed - Pure C inference for perplexity-ai/pplx-embed-v1-0.6b
# Makefile

CC     = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS     = -lm -lpthread

UNAME_S := $(shell uname -s)

# Source files
SRCS = pplx_embed.c \
       qwen_asr_kernels.c \
       qwen_asr_kernels_generic.c \
       qwen_asr_kernels_neon.c \
       qwen_asr_kernels_avx.c \
       qwen_asr_tokenizer.c \
       qwen_asr_safetensors.c

MLX_SRCS = pplx_embed_mlx.c

OBJS     = $(SRCS:.c=.o)
MLX_OBJS = $(MLX_SRCS:.c=.o)
TARGET   = pplx_embed

.PHONY: all blas mlx debug clean help

all: help

help:
	@echo "pplx_embed - Pure C inference for pplx-embed-v1-0.6b"
	@echo ""
	@echo "Build targets:"
	@echo "  make blas     Build with BLAS acceleration (CPU)"
	@echo "  make mlx      Build with Apple MLX GPU backend (recommended on Apple Silicon)"
	@echo "  make debug    Debug build with AddressSanitizer"
	@echo "  make clean    Remove build artifacts"
	@echo ""
	@echo "Usage after build:"
	@echo "  ./pplx_embed -d /path/to/model-dir \"text1\" \"text2\""
	@echo "  ./pplx_embed -d /path/to/model-dir --mlx \"text1\" \"text2\"  (if built with make mlx)"

# =============================================================================
# BLAS build (Apple Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
blas: CFLAGS  = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS  = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas:
	$(MAKE) clean
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"
	@echo ""
	@echo "Built: $(TARGET) (CPU/BLAS)"

# =============================================================================
# MLX build (Apple Silicon GPU - uses mlx-c pure C API)
# Requires: brew install mlx mlx-c
# =============================================================================
MLX_PREFIX  := $(shell brew --prefix mlx 2>/dev/null)
MLXC_PREFIX := $(shell brew --prefix mlx-c 2>/dev/null)

mlx: SRCS += pplx_embed_mlx.c
mlx: OBJS  = $(SRCS:.c=.o) pplx_embed_mlx.o
ifeq ($(UNAME_S),Darwin)
mlx: CFLAGS  = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK -DUSE_MLX \
               -I$(MLXC_PREFIX)/include
mlx: LDFLAGS += -framework Accelerate -framework Metal -framework Foundation \
                -L$(MLXC_PREFIX)/lib -lmlxc \
                -L$(MLX_PREFIX)/lib -lmlx \
                -Wl,-rpath,$(MLX_PREFIX)/lib -Wl,-rpath,$(MLXC_PREFIX)/lib
endif
mlx:
	$(MAKE) clean
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" EXTRA_OBJS="pplx_embed_mlx.o"
	@echo ""
	@echo "Built: $(TARGET) (CPU/BLAS + MLX GPU, use --mlx flag)"

# =============================================================================
# Debug build
# =============================================================================
debug: CFLAGS  = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address
debug: LDFLAGS += -fsanitize=address
debug:
	$(MAKE) clean
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# =============================================================================
# Link
# =============================================================================
$(TARGET): $(OBJS) $(EXTRA_OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# =============================================================================
# Compile rules
# =============================================================================
pplx_embed.o: pplx_embed.c pplx_embed.h qwen_asr_kernels.h qwen_asr_safetensors.h
	$(CC) $(CFLAGS) -c -o $@ $<

pplx_embed_mlx.o: pplx_embed_mlx.c pplx_embed_mlx.h pplx_embed.h qwen_asr_safetensors.h
	$(CC) $(CFLAGS) -c -o $@ $<

main.o: main.c pplx_embed.h qwen_asr_kernels.h qwen_asr_tokenizer.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_asr_kernels.o: qwen_asr_kernels.c qwen_asr_kernels.h qwen_asr_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_asr_kernels_generic.o: qwen_asr_kernels_generic.c qwen_asr_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_asr_kernels_neon.o: qwen_asr_kernels_neon.c qwen_asr_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_asr_kernels_avx.o: qwen_asr_kernels_avx.c qwen_asr_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_asr_tokenizer.o: qwen_asr_tokenizer.c qwen_asr_tokenizer.h qwen_asr_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_asr_safetensors.o: qwen_asr_safetensors.c qwen_asr_safetensors.h
	$(CC) $(CFLAGS) -c -o $@ $<

# =============================================================================
clean:
	rm -f $(OBJS) pplx_embed_mlx.o main.o $(TARGET)

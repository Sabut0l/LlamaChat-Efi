/* =========================================================================
 * Llama2.c  --  Llama-2 transformer inference in the EFI environment.
 * Port of karpathy/llama2.c (run.c + runq.c). Differences from the original:
 *   - no mmap/fopen: the checkpoint and tokenizer arrive as preloaded buffers
 *   - no libm: math from LlamaRuntime (LmExpf/LmSqrtf/LmPowf/...)
 *   - no qsort/bsearch from libc: LmQsort / LmBsearch
 *   - no pthreads/OpenMP: matmul is parallelized over APs via LlamaMp
 *   - no printf: callers render pieces; sscanf hex parse done manually
 *
 * TWO checkpoint formats are autodetected in BuildTransformer:
 *   - legacy fp32 (export.py --version 0): header = raw Config, first int
 *     is dim (small positive number);
 *   - int8 Q8_0  (export.py --version 2): first uint32 is the magic
 *     0x616b3432 ("ak42"), version 2, 256-byte header, fp32 rmsnorm
 *     weights followed by group-quantized tensors (int8 values + one fp32
 *     scale per group of group_size). Matmul then runs in int8 with fp32
 *     accumulation (port of runq.c), activations are quantized on the fly.
 * ========================================================================= */
#include "Llama2.h"
#include "LlamaRuntime.h"
#include "LlamaMp.h"
#include "LlamaDebug.h"

#include <Library/BaseMemoryLib.h>
#include <Library/BaseLib.h>
#include <Library/PrintLib.h>

#define Q8_MAGIC        0x616b3432U   /* "ak42": llama2.c export.py v2 */
#define Q8_VERSION      2
#define Q8_HEADER_BYTES 256

/* =========================================================================
 * Q8_0 group quantization helpers (ports from runq.c)
 * ========================================================================= */
STATIC VOID Dequantize(QuantizedTensor* qx, float* x, UINT64 n, INT32 gs) {
    for (UINT64 i = 0; i < n; i++) {
        x[i] = (float)qx->q[i] * qx->s[i / (UINT64)gs];
    }
}

STATIC VOID Quantize(QuantizedTensor* qx, float* x, INT32 n, INT32 gs) {
    INT32 num_groups = n / gs;
    for (INT32 group = 0; group < num_groups; group++) {
        float* xg = x + (UINTN)group * (UINTN)gs;
        /* find the max absolute value in the current group */
        float wmax = 0.0f;
        for (INT32 i = 0; i < gs; i++) {
            float v = xg[i] < 0.0f ? -xg[i] : xg[i];
            if (v > wmax) wmax = v;
        }
        /* calculate and write the scaling factor */
        float scale = wmax / 127.0f;
        if (scale == 0.0f) scale = 1e-10f;   /* all-zero group */
        qx->s[group] = scale;
        /* calculate and write the quantized values (round to nearest) */
        for (INT32 i = 0; i < gs; i++) {
            float qv = xg[i] / scale;
            qx->q[(UINTN)group * (UINTN)gs + i] = (INT8)(qv >= 0.0f ? qv + 0.5f : qv - 0.5f);
        }
    }
}

/* Map n tensors of size_each quantized values each, laid out back-to-back
 * as [int8 q[size_each]] [float s[size_each/gs]], directly onto the
 * preloaded checkpoint buffer (no copy). Returns NULL on truncation/OOM. */
STATIC QuantizedTensor* InitQuantizedTensors(UINT8** ptr, UINT8* end, INT32 n, UINT64 size_each, INT32 gs) {
    QuantizedTensor* res = (QuantizedTensor*)LmAlloc((UINTN)n * sizeof(QuantizedTensor));
    if (res == NULL) return NULL;
    UINT8* p = *ptr;
    for (INT32 i = 0; i < n; i++) {
        UINT64 need = size_each + (size_each / (UINT64)gs) * sizeof(float);
        if ((UINT64)(end - p) < need) {
            DBG_DEC("InitQuantizedTensors: checkpoint truncated at tensor", (UINT64)i);
            LmFree(res);
            return NULL;
        }
        res[i].q = (INT8*)p;   p += size_each;
        res[i].s = (float*)p;  p += (size_each / (UINT64)gs) * sizeof(float);
    }
    *ptr = p;
    return res;
}

/* =========================================================================
 * Transformer: allocation and initialization
 * ========================================================================= */
STATIC EFI_STATUS MallocRunState(RunState* s, Config* p, INT32 quantized, INT32 gs) {
    INT32 kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x      = (float*)LmCalloc((UINTN)p->dim, sizeof(float));
    s->xb     = (float*)LmCalloc((UINTN)p->dim, sizeof(float));
    s->xb2    = (float*)LmCalloc((UINTN)p->dim, sizeof(float));
    s->hb     = (float*)LmCalloc((UINTN)p->hidden_dim, sizeof(float));
    s->hb2    = (float*)LmCalloc((UINTN)p->hidden_dim, sizeof(float));
    s->q      = (float*)LmCalloc((UINTN)p->dim, sizeof(float));
    s->key_cache   = (float*)LmCalloc((UINTN)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)LmCalloc((UINTN)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att    = (float*)LmCalloc((UINTN)p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float*)LmCalloc((UINTN)p->vocab_size, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        DBG("MallocRunState: allocation FAILED");
        return EFI_OUT_OF_RESOURCES;
    }
    if (quantized) {
        s->xq.q = (INT8*)LmCalloc((UINTN)p->dim, sizeof(INT8));
        s->xq.s = (float*)LmCalloc((UINTN)(p->dim / gs), sizeof(float));
        s->hq.q = (INT8*)LmCalloc((UINTN)p->hidden_dim, sizeof(INT8));
        s->hq.s = (float*)LmCalloc((UINTN)(p->hidden_dim / gs), sizeof(float));
        if (!s->xq.q || !s->xq.s || !s->hq.q || !s->hq.s) {
            DBG("MallocRunState: quantized buffers allocation FAILED");
            return EFI_OUT_OF_RESOURCES;
        }
    }
    return EFI_SUCCESS;
}

STATIC VOID FreeRunState(RunState* s) {
    LmFree(s->x);   LmFree(s->xb);  LmFree(s->xb2);
    LmFree(s->hb);  LmFree(s->hb2); LmFree(s->q);
    LmFree(s->att); LmFree(s->logits);
    LmFree(s->key_cache); LmFree(s->value_cache);
    LmFree(s->xq.q); LmFree(s->xq.s);
    LmFree(s->hq.q); LmFree(s->hq.s);
}

STATIC VOID MemoryMapWeights(TransformerWeights* w, Config* p, float* ptr, INT32 shared_weights) {
    INT32 head_size = p->dim / p->n_heads;
    UINT64 n_layers = (UINT64)p->n_layers;
    w->token_embedding_table = ptr;  ptr += (UINT64)p->vocab_size * p->dim;
    w->rms_att_weight = ptr;         ptr += n_layers * p->dim;
    w->wq = ptr;                     ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;                     ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;                     ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;                     ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;         ptr += n_layers * p->dim;
    w->w1 = ptr;                     ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;                     ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;                     ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;       ptr += p->dim;
    ptr += (UINT64)p->seq_len * head_size / 2;   /* skip legacy freq_cis_real */
    ptr += (UINT64)p->seq_len * head_size / 2;   /* skip legacy freq_cis_imag */
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

/* Предрасчёт RoPE cos/sin: freq(pos, head_dim) зависит только от позиции
 * и индекса пары, поэтому таблица (seq_len * head_size/2) считается один раз
 * при загрузке. Раньше КАЖДЫЙ токен тратил (dim/2)*n_layers вызовов
 * powf+cosf+sinf софт-математики -- 7680 вызовов/токен на 198M-модели. */
STATIC EFI_STATUS FinishBuild(Transformer* t, INT32 quantized, INT32 gs) {
    INT32 head_size = t->config.dim / t->config.n_heads;
    INT32 pairs = head_size / 2;
    UINT64 n = (UINT64)t->config.seq_len * (UINT64)pairs;
    t->rope_cos = (float*)LmAlloc((UINTN)n * sizeof(float));
    t->rope_sin = (float*)LmAlloc((UINTN)n * sizeof(float));
    if (t->rope_cos == NULL || t->rope_sin == NULL) {
        DBG("FinishBuild: RoPE tables allocation FAILED");
        return EFI_OUT_OF_RESOURCES;
    }
    for (INT32 pos = 0; pos < t->config.seq_len; pos++) {
        float* rc = t->rope_cos + (UINT64)pos * (UINT64)pairs;
        float* rs = t->rope_sin + (UINT64)pos * (UINT64)pairs;
        for (INT32 i = 0; i < pairs; i++) {
            float freq = 1.0f / LmPowf(10000.0f, (float)(2 * i) / (float)head_size);
            float val  = (float)pos * freq;
            rc[i] = LmCosf(val);
            rs[i] = LmSinf(val);
        }
    }
    return MallocRunState(&t->state, &t->config, quantized, gs);
}

/* int8 Q8_0 checkpoint (export.py --version 2, runq.c format) */
STATIC EFI_STATUS BuildTransformerQ8(Transformer* t, VOID* checkpoint, UINTN checkpoint_size) {
    UINT8* buf = (UINT8*)checkpoint;
    UINT8* end = buf + checkpoint_size;

    if (checkpoint_size < Q8_HEADER_BYTES) {
        DBG("BuildTransformerQ8: buffer smaller than header");
        return EFI_INVALID_PARAMETER;
    }
    INT32 version = 0;
    CopyMem(&version, buf + 4, sizeof(INT32));
    if (version != Q8_VERSION) {
        DBG_DEC("BuildTransformerQ8: unsupported version", (UINT64)(UINT32)version);
        return EFI_INVALID_PARAMETER;
    }
    CopyMem(&t->config, buf + 8, sizeof(Config));
    UINT8 shared_classifier = buf[8 + sizeof(Config)];
    INT32 gs = 0;
    CopyMem(&gs, buf + 8 + sizeof(Config) + 1, sizeof(INT32));

    Config* p = &t->config;
    if (p->dim <= 0 || p->hidden_dim <= 0 || p->n_layers <= 0 || p->n_heads <= 0
     || p->n_kv_heads <= 0 || p->vocab_size <= 0 || p->seq_len <= 0
     || (p->dim % p->n_heads) != 0
     || gs <= 0 || (p->dim % gs) != 0 || (p->hidden_dim % gs) != 0) {
        DBG("BuildTransformerQ8: implausible config -- wrong model file?");
        return EFI_INVALID_PARAMETER;
    }

    t->quantized  = 1;
    t->group_size = gs;
    t->data       = checkpoint;
    t->file_size  = checkpoint_size;

    TransformerWeights* w = &t->weights;
    UINT64 n_layers = (UINT64)p->n_layers;
    UINT64 kv_dim   = (UINT64)(p->dim * p->n_kv_heads) / p->n_heads;

    /* fp32 rmsnorm weights come first */
    float* fptr = (float*)(buf + Q8_HEADER_BYTES);
    w->rms_att_weight   = fptr;  fptr += n_layers * p->dim;
    w->rms_ffn_weight   = fptr;  fptr += n_layers * p->dim;
    w->rms_final_weight = fptr;  fptr += p->dim;

    /* then the group-quantized tensors, mapped in place */
    UINT8* ptr = (UINT8*)fptr;
    if (ptr > end) {
        DBG("BuildTransformerQ8: truncated after norm weights");
        return EFI_INVALID_PARAMETER;
    }
    w->q_tokens = InitQuantizedTensors(&ptr, end, 1, (UINT64)p->vocab_size * p->dim, gs);
    w->qwq = InitQuantizedTensors(&ptr, end, p->n_layers, (UINT64)p->dim * p->dim, gs);
    w->qwk = InitQuantizedTensors(&ptr, end, p->n_layers, (UINT64)p->dim * kv_dim, gs);
    w->qwv = InitQuantizedTensors(&ptr, end, p->n_layers, (UINT64)p->dim * kv_dim, gs);
    w->qwo = InitQuantizedTensors(&ptr, end, p->n_layers, (UINT64)p->dim * p->dim, gs);
    w->qw1 = InitQuantizedTensors(&ptr, end, p->n_layers, (UINT64)p->dim * p->hidden_dim, gs);
    w->qw2 = InitQuantizedTensors(&ptr, end, p->n_layers, (UINT64)p->hidden_dim * p->dim, gs);
    w->qw3 = InitQuantizedTensors(&ptr, end, p->n_layers, (UINT64)p->dim * p->hidden_dim, gs);
    w->qwcls = shared_classifier
             ? w->q_tokens
             : InitQuantizedTensors(&ptr, end, 1, (UINT64)p->dim * p->vocab_size, gs);
    if (!w->q_tokens || !w->qwq || !w->qwk || !w->qwv || !w->qwo
     || !w->qw1 || !w->qw2 || !w->qw3 || !w->qwcls) {
        DBG("BuildTransformerQ8: weight mapping FAILED (truncated file or OOM)");
        return EFI_INVALID_PARAMETER;
    }

    /* dequantize the token embedding table once: fp32 rows are read at
     * every step, and rmsnorm/rope/attention run in fp32 (as in runq.c) */
    w->token_embedding_table = (float*)LmAlloc((UINTN)p->vocab_size * (UINTN)p->dim * sizeof(float));
    if (w->token_embedding_table == NULL) {
        DBG("BuildTransformerQ8: embedding table allocation FAILED");
        return EFI_OUT_OF_RESOURCES;
    }
    Dequantize(w->q_tokens, w->token_embedding_table, (UINT64)p->vocab_size * p->dim, gs);

    DBG("model format: int8 Q8_0 (v2)");
    DBG_DEC("model dim", p->dim);
    DBG_DEC("model n_layers", p->n_layers);
    DBG_DEC("model n_heads", p->n_heads);
    DBG_DEC("model vocab_size", p->vocab_size);
    DBG_DEC("model seq_len", p->seq_len);
    DBG_DEC("model group_size", (UINT64)(UINT32)gs);
    DBG_DEC("model shared_classifier", (UINT64)shared_classifier);

    return FinishBuild(t, 1, gs);
}

EFI_STATUS BuildTransformer(Transformer* t, VOID* checkpoint, UINTN checkpoint_size) {
    if (checkpoint == NULL || checkpoint_size < sizeof(Config)) {
        DBG("BuildTransformer: bad checkpoint buffer");
        return EFI_INVALID_PARAMETER;
    }
    ZeroMem(&t->weights, sizeof(TransformerWeights));
    ZeroMem(&t->state, sizeof(RunState));
    t->quantized  = 0;
    t->group_size = 1;

    /* format detection: v2 files start with the "ak42" magic; a legacy v0
     * header starts with dim -- a small positive int that can never equal
     * the magic. */
    UINT32 magic = 0;
    CopyMem(&magic, checkpoint, sizeof(UINT32));
    if (magic == Q8_MAGIC) {
        return BuildTransformerQ8(t, checkpoint, checkpoint_size);
    }

    CopyMem(&t->config, checkpoint, sizeof(Config));
    /* negative vocab_size signals unshared classifier weights (run.c trick) */
    INT32 shared_weights = t->config.vocab_size > 0 ? 1 : 0;
    if (t->config.vocab_size < 0) t->config.vocab_size = -t->config.vocab_size;

    if (t->config.dim <= 0 || t->config.n_layers <= 0 || t->config.n_heads <= 0
     || t->config.vocab_size <= 0 || t->config.seq_len <= 0
     || (t->config.dim % t->config.n_heads) != 0) {
        DBG("BuildTransformer: implausible config -- wrong model file?");
        return EFI_INVALID_PARAMETER;
    }

    t->data = checkpoint;
    t->file_size = checkpoint_size;
    float* weights_ptr = (float*)((UINT8*)checkpoint + sizeof(Config));
    MemoryMapWeights(&t->weights, &t->config, weights_ptr, shared_weights);

    DBG("model format: legacy fp32 (v0)");
    DBG_DEC("model dim", t->config.dim);
    DBG_DEC("model n_layers", t->config.n_layers);
    DBG_DEC("model n_heads", t->config.n_heads);
    DBG_DEC("model vocab_size", t->config.vocab_size);
    DBG_DEC("model seq_len", t->config.seq_len);

    return FinishBuild(t, 0, 1);
}

VOID FreeTransformer(Transformer* t) {
    FreeRunState(&t->state);
    LmFree(t->rope_cos); t->rope_cos = NULL;
    LmFree(t->rope_sin); t->rope_sin = NULL;
    if (t->quantized) {
        LmFree(t->weights.token_embedding_table);   /* dequantized copy */
        if (t->weights.qwcls != t->weights.q_tokens) LmFree(t->weights.qwcls);
        LmFree(t->weights.q_tokens);
        LmFree(t->weights.qwq); LmFree(t->weights.qwk);
        LmFree(t->weights.qwv); LmFree(t->weights.qwo);
        LmFree(t->weights.qw1); LmFree(t->weights.qw2); LmFree(t->weights.qw3);
        t->quantized = 0;
    }
    if (t->data != NULL) { LmFree(t->data); t->data = NULL; }
}

/* =========================================================================
 * Neural net blocks (straight ports from run.c; matmul -> LlamaMp MatMul)
 * ========================================================================= */
STATIC VOID RmsNorm(float* o, float* x, float* weight, INT32 size) {
    float ss = 0.0f;
    for (INT32 j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= (float)size;
    ss += 1e-5f;
    ss = 1.0f / LmSqrtf(ss);
    for (INT32 j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

STATIC VOID Softmax(float* x, INT32 size) {
    float max_val = x[0];
    for (INT32 i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (INT32 i = 0; i < size; i++) { x[i] = LmFastExpf(x[i] - max_val); sum += x[i]; }
    float inv = 1.0f / sum;
    for (INT32 i = 0; i < size; i++) x[i] *= inv;
}

STATIC float* ForwardF32(Transformer* transformer, INT32 token, INT32 pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float* x = s->x;
    INT32 dim = p->dim;
    INT32 kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    INT32 kv_mul = p->n_heads / p->n_kv_heads;   /* multiquery sharing factor */
    INT32 hidden_dim = p->hidden_dim;
    INT32 head_size = dim / p->n_heads;

    /* copy the token embedding into x */
    float* content_row = w->token_embedding_table + (UINT64)token * dim;
    CopyMem(x, content_row, (UINTN)dim * sizeof(float));

    for (UINT64 l = 0; l < (UINT64)p->n_layers; l++) {
        /* attention rmsnorm */
        RmsNorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        /* kv cache positions for this layer */
        UINT64 loff = l * (UINT64)p->seq_len * kv_dim;
        s->k = s->key_cache   + loff + (UINT64)pos * kv_dim;
        s->v = s->value_cache + loff + (UINT64)pos * kv_dim;

        /* qkv matmuls */
        MatMul(s->q, s->xb, w->wq + l * dim * dim,    dim, dim);
        MatMul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        MatMul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        /* RoPE: cos/sin из предрасчитанных таблиц (строка по pos).
         * head_dim чётный (шаг 2) -> индекс = head_dim/2. */
        {
            CONST float* rc = transformer->rope_cos + (UINT64)pos * (UINT64)(head_size / 2);
            CONST float* rs = transformer->rope_sin + (UINT64)pos * (UINT64)(head_size / 2);
            for (INT32 i = 0; i < dim; i += 2) {
                INT32 hdi = (i % head_size) >> 1;
                float fcr = rc[hdi];
                float fci = rs[hdi];
                INT32 rotn = i < kv_dim ? 2 : 1;   /* 2 = rotate both q and k */
                for (INT32 v = 0; v < rotn; v++) {
                    float* vec = v == 0 ? s->q : s->k;
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i]     = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }
        }

        /* multihead attention over all timesteps (BSP; small vs matmuls) */
        for (INT32 h = 0; h < p->n_heads; h++) {
            float* q   = s->q + h * head_size;
            float* att = s->att + (UINT64)h * p->seq_len;
            for (INT32 tstep = 0; tstep <= pos; tstep++) {
                float* k = s->key_cache + loff + (UINT64)tstep * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (INT32 i = 0; i < head_size; i++) score += q[i] * k[i];
                att[tstep] = score / LmSqrtf((float)head_size);
            }
            Softmax(att, pos + 1);
            float* xb = s->xb + h * head_size;
            SetMem(xb, (UINTN)head_size * sizeof(float), 0);
            for (INT32 tstep = 0; tstep <= pos; tstep++) {
                float* v = s->value_cache + loff + (UINT64)tstep * kv_dim + (h / kv_mul) * head_size;
                float a = att[tstep];
                for (INT32 i = 0; i < head_size; i++) xb[i] += a * v[i];
            }
        }

        /* final attention matmul + residual */
        MatMul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        for (INT32 i = 0; i < dim; i++) x[i] += s->xb2[i];

        /* ffn rmsnorm */
        RmsNorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        /* SwiGLU: self.w2(F.silu(self.w1(x)) * self.w3(x)) */
        MatMul(s->hb,  s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        MatMul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        for (INT32 i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= 1.0f / (1.0f + LmFastExpf(-val));   /* silu: быстрый exp */
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        MatMul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);
        for (INT32 i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    /* final rmsnorm + classifier */
    RmsNorm(x, x, w->rms_final_weight, dim);
    MatMul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

/* int8 Q8_0 forward pass (port of runq.c): weights stay int8, activations
 * are group-quantized right before each matmul, everything else (rmsnorm,
 * RoPE, attention, silu, residuals) runs in fp32 exactly like ForwardF32. */
STATIC float* ForwardQ8(Transformer* transformer, INT32 token, INT32 pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float* x = s->x;
    INT32 dim = p->dim;
    INT32 kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    INT32 kv_mul = p->n_heads / p->n_kv_heads;   /* multiquery sharing factor */
    INT32 hidden_dim = p->hidden_dim;
    INT32 head_size = dim / p->n_heads;
    INT32 gs = transformer->group_size;

    /* copy the (dequantized) token embedding into x */
    float* content_row = w->token_embedding_table + (UINT64)token * dim;
    CopyMem(x, content_row, (UINTN)dim * sizeof(float));

    for (UINT64 l = 0; l < (UINT64)p->n_layers; l++) {
        /* attention rmsnorm */
        RmsNorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        /* kv cache positions for this layer */
        UINT64 loff = l * (UINT64)p->seq_len * kv_dim;
        s->k = s->key_cache   + loff + (UINT64)pos * kv_dim;
        s->v = s->value_cache + loff + (UINT64)pos * kv_dim;

        /* qkv matmuls: quantize the activation once, reuse for q/k/v */
        Quantize(&s->xq, s->xb, dim, gs);
        MatMulQ8(s->q, s->xq.q, s->xq.s, w->qwq[l].q, w->qwq[l].s, dim, dim, gs);
        MatMulQ8(s->k, s->xq.q, s->xq.s, w->qwk[l].q, w->qwk[l].s, dim, kv_dim, gs);
        MatMulQ8(s->v, s->xq.q, s->xq.s, w->qwv[l].q, w->qwv[l].s, dim, kv_dim, gs);

        /* RoPE: cos/sin из предрасчитанных таблиц (строка по pos).
         * head_dim чётный (шаг 2) -> индекс = head_dim/2. */
        {
            CONST float* rc = transformer->rope_cos + (UINT64)pos * (UINT64)(head_size / 2);
            CONST float* rs = transformer->rope_sin + (UINT64)pos * (UINT64)(head_size / 2);
            for (INT32 i = 0; i < dim; i += 2) {
                INT32 hdi = (i % head_size) >> 1;
                float fcr = rc[hdi];
                float fci = rs[hdi];
                INT32 rotn = i < kv_dim ? 2 : 1;   /* 2 = rotate both q and k */
                for (INT32 v = 0; v < rotn; v++) {
                    float* vec = v == 0 ? s->q : s->k;
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i]     = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }
        }

        /* multihead attention over all timesteps (BSP; small vs matmuls) */
        for (INT32 h = 0; h < p->n_heads; h++) {
            float* q   = s->q + h * head_size;
            float* att = s->att + (UINT64)h * p->seq_len;
            for (INT32 tstep = 0; tstep <= pos; tstep++) {
                float* k = s->key_cache + loff + (UINT64)tstep * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (INT32 i = 0; i < head_size; i++) score += q[i] * k[i];
                att[tstep] = score / LmSqrtf((float)head_size);
            }
            Softmax(att, pos + 1);
            float* xb = s->xb + h * head_size;
            SetMem(xb, (UINTN)head_size * sizeof(float), 0);
            for (INT32 tstep = 0; tstep <= pos; tstep++) {
                float* v = s->value_cache + loff + (UINT64)tstep * kv_dim + (h / kv_mul) * head_size;
                float a = att[tstep];
                for (INT32 i = 0; i < head_size; i++) xb[i] += a * v[i];
            }
        }

        /* final attention matmul + residual */
        Quantize(&s->xq, s->xb, dim, gs);
        MatMulQ8(s->xb2, s->xq.q, s->xq.s, w->qwo[l].q, w->qwo[l].s, dim, dim, gs);
        for (INT32 i = 0; i < dim; i++) x[i] += s->xb2[i];

        /* ffn rmsnorm */
        RmsNorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        /* SwiGLU: self.w2(F.silu(self.w1(x)) * self.w3(x)) */
        Quantize(&s->xq, s->xb, dim, gs);
        MatMulQ8(s->hb,  s->xq.q, s->xq.s, w->qw1[l].q, w->qw1[l].s, dim, hidden_dim, gs);
        MatMulQ8(s->hb2, s->xq.q, s->xq.s, w->qw3[l].q, w->qw3[l].s, dim, hidden_dim, gs);
        for (INT32 i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= 1.0f / (1.0f + LmFastExpf(-val));   /* silu: быстрый exp */
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        Quantize(&s->hq, s->hb, hidden_dim, gs);
        MatMulQ8(s->xb, s->hq.q, s->hq.s, w->qw2[l].q, w->qw2[l].s, hidden_dim, dim, gs);
        for (INT32 i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    /* final rmsnorm + classifier */
    RmsNorm(x, x, w->rms_final_weight, dim);
    Quantize(&s->xq, x, dim, gs);
    MatMulQ8(s->logits, s->xq.q, s->xq.s, w->qwcls[0].q, w->qwcls[0].s, dim, p->vocab_size, gs);
    return s->logits;
}

float* Forward(Transformer* transformer, INT32 token, INT32 pos) {
    if (transformer->quantized) return ForwardQ8(transformer, token, pos);
    return ForwardF32(transformer, token, pos);
}

/* =========================================================================
 * Tokenizer (BPE)
 * ========================================================================= */
STATIC int CompareTokens(CONST VOID* a, CONST VOID* b) {
    return (int)AsciiStrCmp(((CONST TokenIndex*)a)->str, ((CONST TokenIndex*)b)->str);
}

EFI_STATUS BuildTokenizer(Tokenizer* t, VOID* tok_data, UINTN tok_size, INT32 vocab_size) {
    DBG_DEC("BuildTokenizer: input bytes", tok_size);
    DBG_DEC("BuildTokenizer: expected vocab", vocab_size);
    t->vocab_size = vocab_size;
    t->vocab        = (CHAR8**)LmAlloc((UINTN)vocab_size * sizeof(CHAR8*));
    t->vocab_scores = (float*)LmAlloc((UINTN)vocab_size * sizeof(float));
    t->sorted_vocab = NULL;   /* lazily built in Encode */
    if (!t->vocab || !t->vocab_scores) {
        DBG("BuildTokenizer: table allocation FAILED");
        return EFI_OUT_OF_RESOURCES;
    }
    DBG("BuildTokenizer: tables allocated");
    for (INT32 i = 0; i < 256; i++) {
        t->byte_pieces[i * 2]     = (UINT8)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    /* parse the tokenizer.bin byte stream with a cursor */
    UINT8* cur = (UINT8*)tok_data;
    UINT8* end = cur + tok_size;
    if (cur + sizeof(INT32) > end) {
        DBG("BuildTokenizer: missing header");
        return EFI_INVALID_PARAMETER;
    }
    CopyMem(&t->max_token_length, cur, sizeof(INT32)); cur += sizeof(INT32);
    DBG_DEC("BuildTokenizer: max token length", (UINT64)(UINT32)t->max_token_length);

    if (t->max_token_length <= 0 || t->max_token_length > 1024 * 1024) {
        DBG("BuildTokenizer: implausible max token length");
        return EFI_INVALID_PARAMETER;
    }

    for (INT32 i = 0; i < vocab_size; i++) {
        if (cur + sizeof(float) + sizeof(INT32) > end) {
            DBG_DEC("BuildTokenizer: truncated at token", (UINT64)i);
            return EFI_INVALID_PARAMETER;
        }
        CopyMem(&t->vocab_scores[i], cur, sizeof(float)); cur += sizeof(float);
        INT32 len;
        CopyMem(&len, cur, sizeof(INT32)); cur += sizeof(INT32);
        if (len < 0 || (UINTN)len > (UINTN)(end - cur)) {
            DBG_DEC("BuildTokenizer: invalid length at token", (UINT64)i);
            DBG_DEC("BuildTokenizer: invalid length value", (UINT64)(UINT32)len);
            DBG_DEC("BuildTokenizer: bytes remaining", (UINT64)(end - cur));
            return EFI_INVALID_PARAMETER;
        }
        t->vocab[i] = (CHAR8*)LmAlloc((UINTN)len + 1);
        if (!t->vocab[i]) {
            DBG_DEC("BuildTokenizer: token allocation FAILED at", (UINT64)i);
            DBG_DEC("BuildTokenizer: requested bytes", (UINT64)len + 1);
            return EFI_OUT_OF_RESOURCES;
        }
        CopyMem(t->vocab[i], cur, (UINTN)len); cur += len;
        t->vocab[i][len] = '\0';
    }
    DBG_DEC("BuildTokenizer: final stream offset", (UINT64)(cur - (UINT8*)tok_data));
    DBG_DEC("BuildTokenizer: trailing bytes", (UINT64)(end - cur));
    DBG_DEC("BuildTokenizer: tokens loaded", (UINT64)vocab_size);
    return EFI_SUCCESS;
}

VOID FreeTokenizer(Tokenizer* t) {
    for (INT32 i = 0; i < t->vocab_size; i++) LmFree(t->vocab[i]);
    LmFree(t->vocab);
    LmFree(t->vocab_scores);
    LmFree(t->sorted_vocab);
}

/* parse "<0xAB>" -> 0xAB; replaces run.c's sscanf(piece, "<0x%02hhX>", ...) */
STATIC BOOLEAN MatchHexByte(CONST CHAR8* piece, UINT8* out) {
    STATIC CONST CHAR8 hex[] = "0123456789ABCDEF";
    if (piece[0] != '<' || piece[1] != '0' || piece[2] != 'x') return FALSE;
    UINT8 v = 0;
    for (INT32 i = 3; i <= 4; i++) {
        CHAR8 c = piece[i];
        CONST CHAR8* p = hex;
        INT32 d = -1;
        for (INT32 j = 0; j < 16; j++) if (c == p[j] || (c >= 'a' && c <= 'f' && (c - 'a' + 10) == j)) { d = j; break; }
        if (d < 0) return FALSE;
        v = (UINT8)(v * 16 + d);
    }
    if (piece[5] != '>') return FALSE;
    *out = v;
    return TRUE;
}

CHAR8* Decode(Tokenizer* t, INT32 prev_token, INT32 token) {
    if (token < 0 || token >= t->vocab_size) return (CHAR8*)"";
    CHAR8* piece = t->vocab[token];
    /* after BOS(1), sentencepiece strips a leading space */
    if (prev_token == 1 && piece[0] == ' ') piece++;
    UINT8 byte_val;
    if (MatchHexByte(piece, &byte_val)) {
        piece = (CHAR8*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

STATIC INT32 StrLookup(CHAR8* str, TokenIndex* sorted_vocab, INT32 vocab_size) {
    TokenIndex tok = { str, 0 };
    TokenIndex* res = (TokenIndex*)LmBsearch(&tok, sorted_vocab, (UINTN)vocab_size, sizeof(TokenIndex), CompareTokens);
    return res != NULL ? res->id : -1;
}

VOID Encode(Tokenizer* t, CHAR8* text, INT8 bos, INT8 eos, INT32* tokens, INT32* n_tokens) {
    if (text == NULL) { *n_tokens = 0; return; }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex*)LmAlloc((UINTN)t->vocab_size * sizeof(TokenIndex));
        for (INT32 i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id  = i;
        }
        LmQsort(t->sorted_vocab, (UINTN)t->vocab_size, sizeof(TokenIndex), CompareTokens);
    }

    /* buffer for merge candidates (2 tokens max_len each, +1 null, +2 utf8) */
    CHAR8* str_buffer = (CHAR8*)LmAlloc((UINTN)t->max_token_length * 2 + 1 + 2);
    UINTN  str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    /* dummy prefix: sentencepiece prepends " " to non-empty input */
    if (text[0] != '\0') {
        CHAR8 space[2] = { ' ', '\0' };
        INT32 dummy_prefix = StrLookup(space, t->sorted_vocab, t->vocab_size);
        if (dummy_prefix != -1) tokens[(*n_tokens)++] = dummy_prefix;
    }

    /* UTF-8 aware first pass: one token (or byte fallbacks) per codepoint */
    for (CHAR8* c = text; *c != '\0'; c++) {
        if (((*c) & 0xC0) != 0x80) str_len = 0;   /* not a continuation byte */
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if (((*(c + 1)) & 0xC0) == 0x80 && str_len < 4) continue;
        INT32 id = StrLookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            /* byte fallback: +3 because tokens 0..2 are <unk>, <s>, </s> */
            for (UINTN i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (INT32)(UINT8)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    /* merge pass: repeatedly fuse the best-scoring consecutive pair */
    while (1) {
        float best_score = -1e10f;
        INT32 best_id = -1;
        INT32 best_idx = -1;
        for (INT32 i = 0; i < (*n_tokens - 1); i++) {
            AsciiSPrint(str_buffer, (UINTN)t->max_token_length * 2 + 3, "%a%a",
                        t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            INT32 id = StrLookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (INT32 i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;
    LmFree(str_buffer);
}

/* =========================================================================
 * Sampler: argmax / temperature / top-p, xorshift RNG (as in run.c)
 * ========================================================================= */
STATIC UINT32 RandomU32(UINT64* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (UINT32)((*state * 0x2545F4914F6CDD1DULL) >> 32);
}
STATIC float RandomF32(UINT64* state) {
    return (float)(RandomU32(state) >> 8) / 16777216.0f;
}

STATIC INT32 SampleArgmax(float* probabilities, INT32 n) {
    INT32 max_i = 0;
    float max_p = probabilities[0];
    for (INT32 i = 1; i < n; i++) {
        if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; }
    }
    return max_i;
}

STATIC INT32 SampleMult(float* probabilities, INT32 n, float coin) {
    float cdf = 0.0f;
    for (INT32 i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

STATIC int CompareProbIndex(CONST VOID* a, CONST VOID* b) {
    CONST ProbIndex* a_ = (CONST ProbIndex*)a;
    CONST ProbIndex* b_ = (CONST ProbIndex*)b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

STATIC INT32 SampleTopp(float* probabilities, INT32 n, float topp, ProbIndex* probindex, float coin) {
    INT32 n0 = 0;
    float cutoff = (1.0f - topp) / (float)(n - 1);
    for (INT32 i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob  = probabilities[i];
            n0++;
        }
    }
    LmQsort(probindex, (UINTN)n0, sizeof(ProbIndex), CompareProbIndex);

    float cumulative_prob = 0.0f;
    INT32 last_idx = n0 - 1;
    for (INT32 i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) { last_idx = i; break; }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (INT32 i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) return probindex[i].index;
    }
    return probindex[last_idx].index;
}

VOID BuildSampler(Sampler* s, INT32 vocab_size, float temperature, float topp, UINT64 rng_seed) {
    s->vocab_size  = vocab_size;
    s->temperature = temperature;
    s->topp        = topp;
    s->rng_state   = rng_seed != 0 ? rng_seed : 0x123456789ABCDEFULL;
    s->probindex   = (ProbIndex*)LmAlloc((UINTN)vocab_size * sizeof(ProbIndex));
}

VOID FreeSampler(Sampler* s) {
    LmFree(s->probindex);
}

INT32 Sample(Sampler* s, float* logits) {
    INT32 next;
    if (s->temperature == 0.0f) {
        next = SampleArgmax(logits, s->vocab_size);
    } else {
        for (INT32 q = 0; q < s->vocab_size; q++) logits[q] /= s->temperature;
        Softmax(logits, s->vocab_size);
        float coin = RandomF32(&s->rng_state);
        if (s->topp <= 0.0f || s->topp >= 1.0f) {
            next = SampleMult(logits, s->vocab_size, coin);
        } else {
            next = SampleTopp(logits, s->vocab_size, s->topp, s->probindex, coin);
        }
    }
    return next;
}

/* repetition penalty (HF transformers style): вызывать ПОСЛЕ Forward и
 * ДО Sample. Логиты уже встречавшихся токенов делятся на penalty (>0),
 * отрицательные -- умножаются (знак сохраняется). 1.0 -- выключено.
 * У мелких моделей ломает циклы повторов фраз до конца окна. */
VOID ApplyRepetitionPenalty(float* logits, INT32 vocab_size,
                            CONST INT32* recent, INT32 n_recent, float penalty) {
    if (penalty == 1.0f || recent == NULL || n_recent <= 0) return;
    for (INT32 i = 0; i < n_recent; i++) {
        INT32 tok = recent[i];
        if (tok < 0 || tok >= vocab_size) continue;
        if (logits[tok] > 0.0f) logits[tok] /= penalty;
        else                    logits[tok] *= penalty;
    }
}

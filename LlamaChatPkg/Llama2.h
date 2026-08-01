/* =========================================================================
 * Llama2.h  --  Llama-2 transformer inference (port of karpathy/llama2.c
 * run.c + runq.c) to the EFI environment. libc/mmap/printf/pthreads removed;
 * memory comes from a preloaded buffer, math from LlamaRuntime, and matmul
 * from the MP Services worker pool (LlamaMp).
 *
 * Supports TWO checkpoint formats (autodetected in BuildTransformer):
 *   - legacy fp32 (export.py --version 0): header = raw Config;
 *   - int8 Q8_0  (export.py --version 2): magic "ak42", version 2,
 *     group-quantized weights (int8 + fp32 scale per group). ~3.75x smaller
 *     files, quantized matmul ported from runq.c.
 * ========================================================================= */
#ifndef LLAMA2_H
#define LLAMA2_H

#include <Uefi.h>

typedef struct {
    INT32 dim;          /* transformer dimension                         */
    INT32 hidden_dim;   /* for ffn layers                                */
    INT32 n_layers;     /* number of layers                              */
    INT32 n_heads;      /* number of query heads                         */
    INT32 n_kv_heads;   /* number of key/value heads (multiquery)        */
    INT32 vocab_size;   /* vocabulary size (usually 256, byte-level)     */
    INT32 seq_len;      /* max sequence length                           */
} Config;

/* Group-quantized tensor (runq.c): int8 values + one fp32 scale per group
 * of group_size consecutive values. q and s point INTO the preloaded
 * checkpoint buffer for weights, or into LmAlloc'ed buffers for
 * activations. */
typedef struct {
    INT8*  q;   /* quantized values                */
    float* s;   /* scale factors, one per group    */
} QuantizedTensor;

typedef struct {
    /* --- fp32 weights: always present ---------------------------------- */
    float* token_embedding_table;   /* (vocab_size, dim); dequantized copy
                                       when the checkpoint is quantized    */
    float* rms_att_weight;          /* (layer, dim)        */
    float* rms_ffn_weight;          /* (layer, dim)        */
    float* rms_final_weight;        /* (dim,)              */
    /* --- fp32 matmul weights (legacy v0 checkpoints only) --------------- */
    float* wq;                      /* (layer, dim, n_heads*head_size)    */
    float* wk;                      /* (layer, dim, n_kv_heads*head_size) */
    float* wv;                      /* (layer, dim, n_kv_heads*head_size) */
    float* wo;                      /* (layer, n_heads*head_size, dim)    */
    float* w1;                      /* (layer, hidden_dim, dim)           */
    float* w2;                      /* (layer, dim, hidden_dim)           */
    float* w3;                      /* (layer, hidden_dim, dim)           */
    float* wcls;                    /* (optional) classifier weights      */
    /* --- int8 Q8_0 matmul weights (v2 checkpoints only) ------------------ */
    QuantizedTensor* q_tokens;      /* (vocab_size, dim)                  */
    QuantizedTensor* qwq;           /* per layer                          */
    QuantizedTensor* qwk;
    QuantizedTensor* qwv;
    QuantizedTensor* qwo;
    QuantizedTensor* qw1;
    QuantizedTensor* qw2;
    QuantizedTensor* qw3;
    QuantizedTensor* qwcls;         /* == q_tokens when classifier shared */
} TransformerWeights;

typedef struct {
    float* x;    float* xb;   float* xb2;
    float* hb;   float* hb2;
    float* q;    float* k;    float* v;
    float* att;  float* logits;
    float* key_cache;
    float* value_cache;
    /* quantized activation buffers (v2 checkpoints only) */
    QuantizedTensor xq;   /* dim        */
    QuantizedTensor hq;   /* hidden_dim */
} RunState;

typedef struct {
    Config             config;
    TransformerWeights weights;
    RunState           state;
    VOID*              data;        /* preloaded checkpoint buffer        */
    UINTN              file_size;
    INT32              quantized;   /* 1 = int8 Q8_0 (v2), 0 = legacy fp32 */
    INT32              group_size;  /* quantization group size (v2)       */
    float*             rope_cos;    /* предрасчёт RoPE (seq_len * head_size/2) */
    float*             rope_sin;
} Transformer;

typedef struct { CHAR8* str; INT32 id; } TokenIndex;

typedef struct {
    CHAR8**     vocab;
    float*      vocab_scores;
    TokenIndex* sorted_vocab;
    INT32       vocab_size;
    UINT32      max_token_length;
    UINT8       byte_pieces[512];   /* all single-byte strings            */
} Tokenizer;

typedef struct { float prob; INT32 index; } ProbIndex;

typedef struct {
    INT32       vocab_size;
    ProbIndex*  probindex;
    float       temperature;
    float       topp;
    UINT64      rng_state;
} Sampler;

/* ---- transformer ------------------------------------------------------- */
EFI_STATUS BuildTransformer(Transformer* t, VOID* checkpoint, UINTN checkpoint_size);
VOID       FreeTransformer(Transformer* t);
float*     Forward(Transformer* transformer, INT32 token, INT32 pos);

/* ---- tokenizer --------------------------------------------------------- */
EFI_STATUS BuildTokenizer(Tokenizer* t, VOID* tok_data, UINTN tok_size, INT32 vocab_size);
VOID       FreeTokenizer(Tokenizer* t);
CHAR8*     Decode(Tokenizer* t, INT32 prev_token, INT32 token);
VOID       Encode(Tokenizer* t, CHAR8* text, INT8 bos, INT8 eos, INT32* tokens, INT32* n_tokens);

/* ---- sampler ----------------------------------------------------------- */
VOID       BuildSampler(Sampler* s, INT32 vocab_size, float temperature, float topp, UINT64 rng_seed);
VOID       FreeSampler(Sampler* s);
INT32      Sample(Sampler* s, float* logits);

/* repetition penalty (HF style): логиты повторов делятся на penalty (>0),
 * отрицательные -- умножаются. 1.0 = выкл. Ломает циклы повторов.     */
VOID       ApplyRepetitionPenalty(float* logits, INT32 vocab_size,
                                  CONST INT32* recent, INT32 n_recent, float penalty);

#endif /* LLAMA2_H */

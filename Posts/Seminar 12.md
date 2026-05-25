# Mentor's Seminar 12

**Date:** Thursday, May 21, 2026
**Recording:** [Read.ai](https://app.read.ai/analytics/meetings/01KS4Z82S52B6EDZV92CB3QCFW?utm_source=Share_CopyLink)

---

## Part 1. CacheBlend: Selective KV Cache Reuse for Efficient RAG Systems

**Speaker:** Aleksandr Golomazov

### Problem

Modern LLM systems use a **prefill + decode** pipeline. During prefill, the model processes the entire input prompt to build KV caches for all tokens. In RAG systems, retrieved documents are added to the prompt, making it very long and the prefill stage computationally expensive — leading to high **time-to-first-token (TTFT)**.

**KV cache reuse** can mitigate this: pre-compute KV caches for frequently used context chunks (e.g., documents in RAG) and reuse them. However, simple concatenation of independently pre-computed KV caches fails because:
- Key/value representations depend on all previous tokens
- Pre-computed chunks are unaware of each other
- Cross-attention interactions between chunks are lost, degrading quality
- The quality gap grows as the number of retrieved chunks increases

**Alternative approaches and their limitations:**
- **Prefix KV cache:** Recompute KV only for the first chunk (no previous tokens) — still needs to recompute most of the prompt
- **Selective recomputation (existing):** Based on **attention sparsity** — high attention values concentrate on few important tokens. Recomputing only these tokens preserves quality with low cost

### Method: CacheBlend

**Objective:** Quickly update pre-computed KV cache so that the forward attention matrix has minimum L2 deviation from full recomputation.

**Token selection insight:** Tokens with the highest KV deviation on one layer are likely to have the highest deviation on the next layer — attention representations change slowly between layers.

**Algorithm:**
1. Fully recompute KV on the **first layer only** to compute KV deviation for each token
2. Select top $R\%$ of tokens with highest KV deviation
3. Recompute these tokens across all remaining layers
4. **Gradual filtering scheme** (proposed but not practically used): decrease the recomputation ratio across layers ($R_1 > R_2 > \dots$)

**Positional alignment fix:** Pre-computed chunks each start at position 0. Authors use **RoPE (Rotary Positional Embeddings)** to fix positional alignment when merging KV chunks.

**Pipelining:** KV caches are stored on slow devices (CPU RAM / SSD). CacheBlend **pipelines** KV loading and recomputation — while KV for the next layer loads into GPU, the system simultaneously recomputes selected tokens for the current layer. This hides computation cost behind loading latency. Latency stays stable until recomputation ratio exceeds what loading can cover.

**Complete system flow:**
1. User sends query (contains context chunks, e.g., from RAG)
2. Hash each chunk, look up pre-computed KV in cache store
3. If found, retrieve and use cached KV; CacheBlend performs selective recomputation
4. Generate answer and save KV for generated content in cache store

### Experiments

- **Datasets:** Wiki, MusicQA (QA), SQuAD-zoom, Multi-news (summarization)
- **Metrics:** F1 score (QA), ROUGE-L (summarization)
- **Baselines:** Full KV reuse (concatenation, no recomputation), full recomputation, prefix caching

**Results:**
- CacheBlend achieves **much lower TTFT** than full recomputation, with quality **similar to full recomputation**
- Full KV reuse (no recomputation) has low TTFT but poor quality
- CacheBlend is **more stable** at higher request rates (throughput)
- Optimal recomputation ratio: **5–20%**. Higher ratios give diminishing returns

### Discussion Highlights (Ilia Karpov)

- RoPE solves the positional encoding problem when merging independently pre-computed chunks
- Key limitation: the method works best when the query is similar to previous cached queries. If the query changes significantly, different tokens become important and cached attention weights may not be applicable. This makes KV cache reuse fundamentally **query-dependent**
- The question of how close a new query must be to the original for caching to remain effective remains open

---

## Part 2. Nomic Embed V2: Training Sparse Mixture of Experts Text Embedding Models

**Speaker:** Илья Тарасов

### Problem

Modern text embedding models improve by increasing size, but larger models are expensive — especially for **multilingual retrieval** where more parameters are needed. Standard dense transformers activate all parameters for every input. The goal: improve retrieval quality while **reducing active parameters** during inference.

### Method: Nomic Embed V2

First **general-purpose Mixture of Experts (MoE) text embedding model** for multilingual retrieval.

**Architecture:**
- 475M total parameters, only **305M active** during inference
- Each MoE layer has multiple expert networks + a router
- Router selects **top-2 experts** per token based on routing probability
- Only selected experts' parameters are activated

**Expert collapse prevention:** Load balancing loss encourages balanced expert utilization:
$$L_{balance} = \alpha \cdot \sum_i f_i \cdot p_i$$
where $f_i$ = fraction of tokens routed to expert $i$, $p_i$ = average routing probability, $\alpha = 1$.

### Training Pipeline

1. **Long-context adaptation:** Replace absolute positional embeddings with **RoPE** in XLM-RoBERTa (up to 8K tokens)
2. **Masked Language Modeling:** Pre-training on next-token prediction from context
3. **Weakly supervised contrastive pre-training:** On 1.6B filtered multilingual query-document pairs
4. **Hard-negative mining:** Select difficult incorrect examples for contrastive training (e.g., for "what is photosynthesis?" — easy negative: football scores; hard negative: cellular respiration)
5. **Contrastive fine-tuning + Matryoshka representation learning:** InfoNCE loss maximizes similarity for correct query-document pairs and minimizes for negatives. Matryoshka training ensures embeddings remain effective when **truncated to smaller dimensions**

### Experiments

- **Benchmarks:** BEIR (English retrieval), MIRACL (multilingual retrieval), GLUE, XTREME (language understanding)
- **Main metric:** NDCG@10 (ranking quality in top 10)
- **Key ablation findings:**
  - MoE consistently outperforms dense baseline with similar active parameter count
  - Larger batch size improves MoE training quality
  - Converting **only some layers** to MoE performs better than converting all layers (partial upcycling balances capacity and optimization stability)
  - Strong hard-negative mining further improves quality

**Results:** 52.86 on BEIR, 65.8 on MIRACL — competitive with models nearly twice as large.

### Limitations & Discussion

- Largest dense multilingual models still outperform on some tasks
- MoE training is more complex and sensitive to optimization settings
- Expert specialization (language, domain, context length) is **not analyzed** — unclear what each expert actually learns
- Hard-negative mining procedure does not control similarity between the hard negative and the original true positive — a paper close in score to the correct answer might also be relevant

### Reproducibility

Strong: code, model checkpoints, evaluation scripts, and training hyperparameters are all released. Entire pipeline trained on publicly available data.

---

## Part 3. Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models

**Speaker:** Abylkhaiyr (Aripkhanov)

### Problem

In RAG and information retrieval, long documents are split into **chunks**, each encoded into an embedding vector. The standard **naive chunking** approach:
1. Split document into chunks
2. Encode each chunk independently

**The lost context problem:** A chunk loses surrounding context. Example: "Berlin is the capital of Germany" in one chunk and "The city has over 3.85M inhabitants" in another — without seeing "Berlin," the model cannot resolve "the city" in the second chunk. Sentences referring to Berlin without containing the word "Berlin" get lower similarity to a "Berlin" query under naive chunking.

### Method: Late Chunking

**Core idea:** Let the model see the full document context **before** chunking.

1. Pass the **full document** (or large portion) into a long-context embedding model
2. Get contextual token embeddings for the entire document
3. **After** encoding, split the token sequence into chunks based on a chosen strategy (fixed size, sentence-based, semantic)
4. Apply **mean pooling** over token embeddings within each chunk boundary

Chunking happens "late" — after the model has already processed the broader context.

### Long Late Chunking

Long-context models have limits (e.g., ~8K tokens). For very long documents:
1. Split document into overlapping **macro-chunks** that fit the context window
2. Apply late chunking inside each macro-chunk
3. Overlap ensures boundary information is preserved

### Span Pooling (Optional Training)

Standard embedding models are trained to represent whole texts. For retrieval, the goal is often to find a specific span containing the answer. **Span pooling training:**
- Training data contains query, document, and annotated relevant span
- Model encodes the whole document, but the final embedding is computed **only from tokens in the relevant span**
- Improves late chunking performance by teaching the model to focus on relevant regions

### Experiments

- **Benchmark:** BEIR (NDCG@10)
- **Chunking strategies tested:** fixed-size, sentence boundaries, semantic sentence boundaries
- **Results:** Late chunking almost always outperforms naive chunking
  - Sentence boundaries: +3.63% relative improvement
  - Fixed-size: +3.46%
  - Semantic sentences: +2.70%

**Limitations:**
- Most useful when chunk meaning depends on surrounding text (pronouns, references like "the city," "this approach")
- On synthetic tasks where important information is not connected to surrounding context, naive chunking can perform similarly or better
- Not a universal solution, but very useful for documents with real semantic connections between parts
- No mandatory retraining required, works with different long-context embedding models

### Discussion Highlights (Ilia Karpov)

- Late chunking is especially relevant for retrieval tasks
- Mentioned modern approaches with document-level embeddings (e.g., DPS embeddings, maximum document-level embeddings)
- Discussed applicability to summarized documentation

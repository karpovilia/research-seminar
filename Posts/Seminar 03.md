# Mentor's Seminar 03

**Date:** Wednesday, Apr 15, 2026
**Recording:** [Read.ai](https://app.read.ai/analytics/meetings/01KP8M9HK4B667VKJ1NAQJRTQ2?utm_source=Share_CopyLink)
---

## Part 1: GRAG -- Graph Retrieval Augmented Generation

**Speaker:** Shahab Azmi

A paper from Emory University that extends standard RAG to handle connected documents -- citations, social media graphs, and knowledge graphs -- by retrieving subgraphs rather than individual documents.

### Problem and motivation

- **Standard RAG treats each document independently**, but real documents are interconnected: research papers cite each other, social media posts reply to each other, knowledge graphs link entities. Ignoring these connections means losing critical context.
- **Two fundamental challenges** in Graph RAG:
  1. *Retrieval* -- finding the best subgraph efficiently; searching all possible subgraphs is NP-hard.
  2. *Generation* -- providing the LLM with both text content and graph structure in a usable form.

### Retrieval: divide and conquer with soft pruning

- **Identify important nodes**, then retrieve **ego graphs** (neighborhoods centered on those nodes).
- **Rank ego graphs** by cosine similarity to the query.
- **Soft pruning** -- learns a scaling factor for each node and edge; nodes far from the query in meaning get a factor near zero, effectively removing them. The scaling factor is reused during generation to control the graph attention network.
- **Merge pruned ego graphs** into the optimal subgraph. The entire process runs in **linear time**, making it practical for large graphs.

### Generation: dual-view prompting

- **Text view (hard prompt)** -- converts the graph into a nested indented list using BFS-based tree traversal, fed directly as a text prompt to the LLM.
- **Graph view (soft prompt)** -- a graph attention network (GAT) encodes the graph topology, controlled by the same scaling factors from soft pruning. An MLP aligns graph embeddings to the LLM's text space.
- The LLM receives **both views together** and generates the final answer.

### Datasets and evaluation

- **WebQSP** -- large dataset: ~13,000 nodes, ~10,000 tokens per graph on average.
- **ExplaGraphs** -- smaller dataset: ~5 nodes per graph, focused on commonsense reasoning.
- Base LLM: **LLaMA2-7B** (frozen, not fine-tuned).
- GRAG with frozen LLM achieves **72% hit@1 on WebQSP** and **92% accuracy on ExplaGraphs** -- 74% and 172% improvement over plain LLM respectively. Frozen LLM + GRAG **outperforms fine-tuned models**.

### Ablation findings

- Removing the graph encoder (soft prompts) drops hit@1 from 0.72 to 0.58.
- Removing soft pruning makes performance worse than using no graph retrieval at all -- irrelevant information confuses the LLM.
- Removing text descriptions causes a ~38% drop -- the actual words of nodes and edges are crucial.
- **Two-hop ego graphs** perform best; adding more hops introduces irrelevant information. However, using more subgraphs gives more robust generation with smaller variance.
- Training on the larger WebQSP dataset improves ExplaGraphs accuracy by **33.77%** (transfer learning works better from large to small).
- GRAG reduces hallucinations significantly: **79% valid references** vs. 71% for GRAG-Retriever alone and 62% for MiniLM.

### Limitations and critique

- **No error analysis** -- the paper does not classify questions by difficulty or analyze where the model fails. Simple questions (e.g., "Who is Justin Bieber's brother?") may be answered correctly by standard RAG, so the uplift from graph retrieval on WebQSP is unclear.
- **WebQSP is a poor fit for Graph RAG evaluation** -- it contains mostly simple, single-hop questions where key information is already in the query. Graph RAG is most useful for domain-specific or multi-hop reasoning questions.
- **No ablation on the number of hops** -- the paper uses two-hop ego graphs but does not justify why three or more hops were not explored.
- **No discussion of graph density effects** -- real knowledge graphs are very dense (e.g., public figures connected to millions of nodes), which may significantly affect community detection and retrieval quality.
- **Evaluation metric inconsistency** -- hit@1 (WebQSP) and accuracy (ExplaGraphs) are computed differently but presented without clarification, causing confusion.
- **Speaker did not attempt to reproduce** the results from the publicly available GitHub repository.

### Discussion highlights

- A 2025 paper showed that **Graph RAG is not a silver bullet** -- classical RAG is efficient for simple lookups; graphs only help with complex, multi-hop, domain-specific reasoning. Evaluation should weight questions by difficulty.
- The question of whether to retrieve **communities** (as GRAG does) vs. **reasoning paths** (as ThinkOnGraph does) remains open, and graph density likely affects which approach works better.
- The dual-view (hard prompt + soft prompt) design is elegant but the paper lacks depth in analyzing **when** and **why** each view contributes.

---

## Part 2: FastRAG -- Retrieval Over Augmented Generation for Semi-Structured Data

**Speaker:** Mudassir Iftikhar

A paper proposing a pipeline for efficiently processing technical semi-structured data -- network logs and configuration files -- by combining intelligent chunk sampling, schema learning, and hybrid retrieval.

### Problem and motivation

- **Network systems generate large volumes of semi-structured data** (logs, configs) that is hard to process with standard RAG.
- Traditional RAG either processes everything (high cost and latency) or fails to capture domain-specific structure.
- **Vector RAG loses context** (log entries lack clean text structure); **graph RAG struggles with precise queries** on semi-structured data.

### Pipeline architecture

The pipeline operates in two parallel tracks:

#### Track A: Chunk sampling and schema learning

- **Chunk sampling** -- intelligently selects representative data chunks rather than processing everything:
  - Preprocessing: remove punctuation, tokenize.
  - Compute **TF-IDF vectors** and **Shannon entropy** for each chunk.
  - Alternatively select chunks that maximize term coverage while introducing maximum diversity.
- **Schema learning** -- an LLM (GPT-4) analyzes sampled chunks and automatically generates a **JSON schema** describing the data structure.
- **Script learning** -- the LLM generates **Python scripts** to parse source data and extract entities according to the learned schema. This makes the system adaptable and reduces manual effort.

#### Track B: Hybrid retrieval

Four retrieval strategies are available:
  1. **Knowledge graph querying** -- builds a KG from extracted entities and relationships for relational queries.
  2. **Text search** -- LLM generates optimized queries for direct text retrieval.
  3. **Combined retrieval** -- runs KG and text search in parallel, then synthesizes results.
  4. **Hybrid retrieval** -- merges KG and text features into a single retrieval pipeline.

### Experimental setup and results

- **Datasets:** network logs (~200 lines) and configuration files (~2,100 lines) -- very small scale.
- **Base model:** GPT-4.
- **Evaluation metrics:** coverage ratio, QA accuracy, time, and cost.

| Metric | FastRAG (logs) | GraphRAG (logs) | FastRAG (configs) | GraphRAG (configs) |
|---|---|---|---|---|
| Time | ~4 min | ~40 min | -- | -- |
| Cost | ~$0.90 | ~$6.00 | -- | -- |

- FastRAG achieves roughly **90% reduction in time and cost** compared to GraphRAG.
- The authors report improved accuracy but do not provide detailed per-question results.

### Main contributions

- **Efficient chunk sampling** using entropy-based selection -- avoids processing the entire dataset.
- **Schema and script learning** -- automatic discovery of data structure from samples, reducing manual effort and enabling adaptation to new log formats.
- **Hybrid retrieval** combining knowledge graph and text search for comprehensive answers.

### Limitations and critique

- **Very small dataset** -- only ~2,100 lines of configuration data and ~200 lines of logs; results may not generalize to production-scale systems.
- **No public code or repository** -- the authors provide only the algorithm description, making reproducibility impossible.
- **No detailed accuracy analysis** -- the paper shows time/cost improvements but does not clearly demonstrate whether FastRAG enables answering questions that were previously unanswerable, or merely saves compute for the same quality.
- **Weak relationship extraction** -- the knowledge graph construction is acknowledged as not fully optimized.
- **Heavy LLM dependency** -- the entire schema learning and script generation pipeline relies on GPT-4 quality; no experiments with smaller or open models.
- **No labeled benchmark** -- evaluating on real-world logs requires correct ground-truth answers, which are hard to obtain at scale.

### Discussion highlights

- The **idea is considered brilliant** -- structuring logs as a graph of virtual machines, services, and endpoints is a natural and powerful approach that all IT companies would benefit from.
- A parallel can be drawn to **section-aware retrieval in academic papers** -- finding relevant information in the Methods section is far more valuable than in the Literature Review, and the same logic applies to log structure.
- The mentor suggested that rather than building the domain model **from scratch** using log messages alone, it would be more effective to leverage **existing system documentation and source code** to discover entities, endpoints, and service interactions -- then enrich the graph with log-derived information.
- The paper is seen as a promising **early step** that could lead to significant practical impact if developed further with larger datasets, proper benchmarks, and integration of existing system knowledge.

---


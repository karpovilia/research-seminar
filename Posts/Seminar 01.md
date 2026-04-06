# Research Seminar 01

**Date:** Thursday, Apr 2, 2026
**Recording:** [Read.ai](https://app.read.ai/analytics/meetings/01KN6VKSDP3Y9CMJGVCSVG15SH?utm_source=Share_CopyLink)

---

## Part 1: Introduction to Network Science and Graph Neural Networks

**Speaker:** Ilia Karpov

A crash course covering the foundations needed to understand the papers we will discuss throughout the course.

### Key concepts

- **Power-law distribution** -- most real-world networks (citations, social graphs) follow a heavy-tailed degree distribution: a few nodes have thousands of connections while most have very few.
- **Triadic closure** -- if A knows B and A knows C, then B and C are likely to become connected over time.
- **Small-world phenomenon** -- Milgram's experiment (1967) showed that a random person in the US can be reached in about 4--6 hops, demonstrating that real networks have surprisingly short path lengths.

### From graphs to embeddings

- **DeepWalk / Node2Vec** -- the first scalable approach: linearize the graph via random walks, then apply Word2Vec-style training on the resulting node sequences. Simple, effective, and much cheaper than spectral methods.
- **Message Passing Framework** -- the unifying idea behind all modern GNNs. Each node aggregates messages from its neighbors using a permutation-invariant function (sum, mean, max, attention), then updates its own hidden state. Repeating this for K layers lets each node "see" its K-hop neighborhood.
- **GCN, GraphSAGE, GAT** -- concrete instantiations of the message passing framework with different choices for message, aggregation, and update functions.

### GNNs vs Transformers

- GNNs are designed for **large, sparse** graphs (millions of nodes, few edges per node).
- Transformers operate on **small, dense** attention matrices (all tokens attend to all tokens).
- A 2025 paper formally proved that multi-layer message passing and multi-head self-attention are mathematically equivalent -- the difference is the sparsity pattern.

### Current frontiers

- **Temporal GNNs** -- adding the time dimension to graph learning; the main research direction covered in this course.
- **Graph RAG** -- using graph structures for retrieval-augmented generation, especially effective in domain-specific settings (medical, legal) where general-purpose LLMs struggle with terminology.

---

## Part 2: RAG in Machine Translation for Domain-Specific Texts

**Speaker:** Maria Sukhareva (guest)

An example of a seminar-style talk reviewing four recent papers on applying retrieval-augmented generation to machine translation, with a focus on terminology accuracy.

### Paper 1: Terminology-Constrained Translation (WMT 2025)

- **Task:** translate IT documentation (EN -> DE/ES/RU) while following glossary terms.
- **Problem:** models tend to ignore glossary translations.
- **Methods:**
  - *Code switching* -- replace source terms with target-language translations before feeding to the model.
  - *LoRA fine-tuning* of EuroLLM (9B) on 200K parallel pairs with automatic code switching via FastAlign.
  - *Pareto optimal decoding* -- generate 100 translation candidates, evaluate on MT quality and terminology success rate (TSR), select from the Pareto front.
- **Result:** near-perfect TSR in Russian and Spanish; improvements in general MT quality as well.

### Paper 2: Neologism Translation with Agentic RAG

- **Task:** translate sentences containing neologisms (newly coined words).
- **Architecture:** an agentic loop where the model decides when it needs help, searches Wiktionary via `<search>` tags, and forms a final translation -- up to 3 retrieval rounds per sentence.
- **Training:** GRPO reinforcement learning with rewards for neologism presence, MT quality, and output format.
- **Dataset:** NeoDataset, automatically compiled from Wiktionary entries tagged as neologisms.
- **Result:** beats all baselines including dedicated Chinese MT models, even in human evaluation.

### Paper 3: Glossaries + Translation Memory via Contrastive Ensembling

- **Task:** integrate two knowledge sources used by professional translators -- glossaries (terms) and translation memory (fuzzy matches from past translations).
- **Method:** fine-tune separate sub-1B models on terms only and fuzzy matches only, then combine via *contrastive ensembling* -- at each decoding step, use the token from whichever model is more confident.
- **Result:** competitive with GEMMA-12B despite being 12x smaller; the ensembling approach successfully leverages both knowledge sources.

### Paper 4: Wikipedia Context for Translation (WeChat AI)

- **Task:** translate the first sentence of Wikipedia articles (surprisingly hard due to dense terminology and named entities).
- **Method:** feed the model additional paragraphs from the same Wikipedia page as context; train with three auxiliary tasks:
  1. Cross-lingual information completion (code-switched summaries).
  2. Self-generated document utilization.
  3. Cross-lingual document relevance classification.
- **Result:** even context from a third language (neither source nor target) improves translation quality after training.

---

## Discussion highlights

- How to adapt Wikipedia-based RAG to domains with no Wikipedia coverage? Possible directions: synthetic paragraph generation via large models, distillation.
- Trade-off between using large models (GPT-4/5) directly vs. distilling their knowledge into smaller, specialized models.



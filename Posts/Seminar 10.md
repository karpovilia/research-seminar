# Research Seminar 10

**Date:** Thursday, May 14, 2026
**Recording:** [Read.ai](https://app.read.ai/analytics/meetings/01KRJZYNK95K1XQS59K7BZ0A1W?utm_source=Share_CopyLink)

---

## Part 1: TAMI -- Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction

**Speaker:** Aleksandr Mandrov

A paper (NeurIPS 2025) proposing TAMI, a framework that addresses two fundamental problems in temporal graph learning: improper temporal encoding of power-law distributed interactions, and the forgetting of old interactions in aggregation functions. TAMI is designed as a plug-in module compatible with a wide range of existing methods.

### Temporal graphs and link prediction

- **Temporal graph:** a graph where edges appear at discrete time steps. At different moments, both the set of vertices and edges may change.
- **Link prediction task:** given the graph history up to time t, predict the probability of an edge between vertices u and v at time t'.
- **Real-world examples:** music streaming services (user listens to song = interaction), culinary preferences (family meal patterns over time).

### Standard approach

1. Compute **temporal embeddings** for each time step using a time encoding function (typically cosine of delta-t scaled by learnable parameters alpha and beta).
2. Aggregate neighbor information via an **aggregation function** -- typically the most recent interaction only (last k interactions).
3. Pass the result through an **MLP** to produce a link probability.

### Two identified problems

**Problem 1: Power-law distribution of temporal interactions.**
- Interaction intervals (delta-t) follow a power-law (Pareto) distribution: most events are recent, few are distant.
- Standard cosine time encoding compresses recent events (small delta-t) into a narrow range, while distant events (large delta-t from the tail) produce disproportionately large values. This creates an imbalance in how time is represented.
- Example: a family eats fast food most days, but eats turkey on Thanksgiving once a year. If only recent interactions are considered, the model will predict fast food for Thanksgiving. The distant-but-relevant seasonal pattern is lost.

**Problem 2: Forgetting of old interactions.**
- Most aggregation functions use only the k most recent interactions.
- Random walk methods also bias toward recent events because recent interactions are more numerous (power-law), so walks are more likely to sample them.
- Rare but important events from the distant past are systematically excluded.

### TAMI: two new modules

**Module 1: LTE (Logarithmic Time Encoding)**
- Replace delta-t with log(delta-t) in the time encoding function.
- Logarithmic compression normalizes the power-law distribution, producing more uniform values across both recent and distant events.
- **Theoretical justification:** the authors prove that if delta-t follows a Pareto distribution with parameter alpha > 3, then after log transformation the third standardized moment drops below 2, while with raw delta-t it exceeds 2. This means the log-transformed distribution has lighter tails and is better behaved for learning.

**Module 2: Link History Aggregation (memory mechanism)**
- Instead of simply selecting the k most recent interactions, maintain a **running memory** of edge embeddings.
- At each time step, the current edge embedding is computed as: MLP(node_u, node_v), combined with the previous embedding scaled by a **forgetting coefficient** gamma.
- Each aggregated vector already contains information from all prior interactions (through the recurrence), so even the k most recent vectors carry historical context.
- This creates an analogy to **memory** -- old interactions are preserved (with decay) rather than discarded.

### Framework design

- TAMI is **not a standalone method** -- it is a drop-in framework that modifies the time encoding and aggregation steps of existing temporal graph models.
- The base architecture (message passing, MLP, etc.) remains unchanged.
- Both modules can be applied independently or together; applying both yields the largest improvement.

### Experiments and results

- **13 temporal graph datasets** (3 standard TG benchmarks + 10 additional).
- **8-10 baseline methods** including GraphMixer, DyGFormer, TGAT, TGN, and others.
- **Metrics:** Average Precision; train/val/test split 75/15/10 with chronological ordering.
- **Negative sampling:** varying ratios of negative-to-positive examples (1:1 to 50:1).

**Key results:**
- **GraphMixer + TAMI:** +2-16% improvement across datasets. Substantial gains.
- **DyGFormer + TAMI:** +2% improvement on most datasets. More modest (DyGFormer already has sophisticated temporal handling).
- **DyGFormer rose from 6th to 2nd place** on TGBL benchmark after applying TAMI.
- **Ablation:** both modules contribute; Link History Aggregation provides the larger individual gain. Applying both together yields the best results.
- **Negative sampling:** the more negative examples relative to positive, the larger the relative improvement from TAMI (10-60% quality increase as ratio goes from 1:1 to 50:1).

**Speaker's reproduction:**
- Tested on one method and one synthetic dataset in Google Colab.
- Confirmed that TAMI modules improve quality on both train and test sets.
- Shared the Colab notebook via Telegram.

### Limitations and critique

- **Power-law distribution shown for one dataset only** -- the authors provide a graph of delta-t distribution on a single dataset. While they claim this holds generally, no multi-dataset visualization is provided.
- **Time granularity not discussed** -- the log encoding treats hours, days, and months identically after transformation. Different granularities may require different encoding strategies. A multi-scale approach (separate low-frequency and high-frequency components, similar to knowledge graph approaches) could be more effective.
- **No comparison with snapshot-based methods** -- it remains unclear whether a simple periodic snapshot approach (e.g., weekly or monthly aggregated graphs) would achieve comparable results with less complexity.
- **Computational overhead** of the memory mechanism (recurrent aggregation at every edge) is not analyzed.
- **No analysis of forgetting coefficient gamma** -- how sensitive are results to the choice of gamma? Is it tuned per dataset or fixed?
- **Speaker's reproduction was limited** -- one method, one synthetic dataset. No GitHub link; code only in a Colab notebook.

### Discussion highlights

- **Seasonal patterns and granularity:** the mentor suggested that the log encoding conflates different time scales. For minutes-level interactions (e.g.,高频 user actions), the log function compresses too aggressively; for monthly patterns (seasonal events), it may not compress enough. A multi-scale or adaptive approach could help.
- **Mixture of experts for time scales:** the mentor proposed training two models with different temporal granularity and using a gating mechanism (ensemble) to select the appropriate model based on the interaction pattern -- one for frequent short-term events, another for rare long-term seasonal patterns.
- **Snapshot-based comparison:** the mentor raised the open question of whether quantizing time into snapshots and training a simpler model could achieve comparable results, especially for small graphs with few changes per snapshot. The hypothesis is that simpler models may overfit less, while the complexity of TAMI may only pay off on large, dense temporal graphs with many changes.
- **The mentor noted the power-law claim aligns with broader observations** that snapshot-based methods tend to underperform event-based methods, but the question of how much complexity is justified remains open.

---

## Announcements

- The TIDFormer speaker was absent; only one presentation was delivered.
- The mentor encouraged anyone interested in graph research to reach out.

# Research Seminar 06


**Date:** Wednesday, Apr 29, 2026
**Recording:** [Read.ai](https://app.read.ai/analytics/meetings/01KPWW008XKHM22D7P1F8EJ3GZ?utm_source=Share_CopyLink)
---

## Part 1: TG-Talker -- Adapting LLMs for Temporal Graph Link Prediction

**Speaker:** Нусратуллина Марьям

A paper (AAAI 2025) proposing TG-Talker, a framework that adapts large language models for temporal graph link prediction via in-context learning, eliminating the need to train task-specific models from scratch.

### Problem and motivation

- **Temporal graph neural networks (T-GNNs)** like TGAT, TGN, and TGCN learn from scratch for every new dataset, require expensive training, and cannot provide text explanations for their predictions.
- **Heuristic-based approaches** exist but also lack explanatory capability.
- **LLMs** offer in-context learning (adapt without parameter updates) and can generate human-readable explanations, but prior work only explored LLMs with static graphs or small synthetic temporal graphs.
- **Gap:** no framework for applying LLMs to real-world temporal graphs.

### Task formulation

- **Link prediction:** given a temporal graph up to time T and a query (source node s, time t'), predict the destination node.
- **Link explanation:** given the graph, the query, and a predicted answer, generate a natural-language explanation of why the prediction was made.
- The LLM is treated as an interface: discrete tokens in, discrete tokens out. A graph encoding function converts graph data into the token space.

### TG-Talker architecture: four components

1. **Background set** -- threshold edges before the prediction moment (default: 300 edges) that provide structural context.
2. **Example set** -- five question-answer pairs demonstrating the task format, enabling in-context learning.
3. **Query set** -- the current link prediction question encoded in tokens.
4. **Temporal neighbors** -- the M most recent neighbors for the node of interest, providing recent local context.

The background set and temporal neighbors are needed because LLMs have fixed maximum input length and cannot ingest an entire real-world graph.

### Evaluation methodology (MRR)

- Temporal GNNs output a score for each candidate node; LLMs output only one predicted node (no scoring).
- For fair comparison: if LLM predicts correctly, rank = 1 (score > all 20 negative candidates). If incorrect, rank = 21 (score < all).
- **MRR (Mean Reciprocal Rank):** for each test edge, reciprocal rank = 1/rank, averaged across all test edges.
- Negative candidates: 20 per test edge (10 historical, 10 random-but-possible nodes that respect graph structure).

### Results

- **T-TGNNs (TNCN, TGAT) still outperform TG-Talker** on most datasets.
- **TG-Talker with Qwen is competitive** and approaches T-GNN performance on some benchmarks.
- Five datasets tested: three bipartite, two user-to-user email.
- **Ablation:** removing temporal neighbors drastically degrades performance; increasing the number of neighbors improves results, confirming their importance.
- **Explanation results:** LLaMA and GPT-4 generated textual explanations, but no formal quality evaluation of explanation accuracy was performed. Hallucinations were observed.

### Limitations and critique

- **Context window limitation** -- LLMs cannot ingest large-scale temporal graphs (millions of edges), limiting practical applicability.
- **No explanation quality metric** -- explanations are generated but never evaluated for correctness or usefulness.
- **Inconsistent hyperparameters** -- the paper says background set size is 300, but the code uses 1000. Several parameter choices lack justification.
- **Speaker did not attempt reproduction** -- no experiments were run to verify the results, despite having access to open-source LLMs.

### Discussion highlights

- **Mentor's practical test:** TG-Talker was tested on a corporate temporal graph task (predicting node features). The result was dramatically worse than traditional methods: TG-Talker achieved ~0.18--0.19 Gini vs. ~0.51--0.52 for T-GNN baselines. The mentor attributed this to: (1) framing classification tasks as link prediction is suboptimal, (2) LLMs received no fine-tuning, and (3) temporal graphs create wide-and-sparse contexts that LLMs handle poorly.
- **Cherry-picked datasets:** a colleague who tested TG-Talker on their own data found that the paper's chosen datasets may have been selected to boost the method's reported metrics.
- The mentor emphasized that **reproduction attempts are a required part of the seminar grade** and offered computational resources and API tokens to students who ask for help.
- The mentor noted that OpenAI batch queries could serve as a low-cost way to partially reproduce the approach.

---

## Announcements

- The second speaker (Egor) was absent, so the seminar ended early.
- The mentor reminded all speakers that attempting to reproduce results -- even partially, even with small models or API-based services -- is a graded requirement.

# Mentor's Seminar 11

**Date:** Thursday, May 20, 2026
**Recording:** [Read.ai](https://app.read.ai/analytics/meetings/01KS2Y4JQP7HACW5QXZF852WRY?utm_source=Share_CopyLink)

---

## Part 1. DG Explainer: Explaining Dynamic GNN Predictions via Relevance Backpropagation

**Speaker:** София (Safiya)

### Problem

Dynamic Graph Neural Networks (Dynamic GNNs) are successful in traffic forecasting, social network analysis, and epidemic modeling, but remain **black boxes** — their predictions cannot be trusted in critical applications without explanations. Existing GNN explanation methods fall into three categories (approximation-based, perturbation-based, gradient-based), and **all of them ignore temporal modules** (RNN, GRU, LSTM). Applying static methods to dynamic graphs loses temporal context. The goal is to compute the **relevance** (importance with positive/negative sign) of each feature of each node at each time step — indicating contribution to the model's prediction.

### Method: DG Explainer

Based on **Layer-wise Relevance Propagation (LRP)**, which backpropagates importance scores from output to input. Two stages:

1. **Stage 1 — Temporal backpropagation:** Propagate relevance backwards in time through the GRU model, from the last time step to the first. Computes the relevance of each previous hidden state $h_{t-1}$ and the relevance of the GNN input $x_t$ at each step. Uses the **epsilon rule** for numerical stability. The GRU gates ($z$ and $n$) connect the relevances.

2. **Stage 2 — Spatial backpropagation:** Propagate relevance backwards through the GNN layer by layer (from last layer $L$ to layer 0), applying the epsilon rule for convolution. Uses a **normalized adjacency matrix** accounting for node degrees. The result is the relevance of the original input features.

Key properties:
- First application of LRP to both temporal and spatial modules **simultaneously**
- No retraining, no surrogate models, no optimization — computed directly
- Output: relevance of each feature, for each node, at each time step

### Experiments

- **Datasets:** 6 real-world datasets (Reddit, Core, Facebook Collab, etc.)
- **Baselines:** 8 methods including both static and specialized dynamic ones
- **Metrics:** Fidelity, stability, and two others

**Results:**
- DG Explainer achieves **best or second-best** results on all datasets in terms of **fidelity**
- In terms of **stability**, competitive but slightly behind simpler methods like sensitivity analysis on some datasets
- **Qualitative analysis** on PMC04 traffic networks: DG Explainer shows meaningful spatial patterns (strongly positive, strongly negative, and intermediate nodes), while baseline methods produce uninformative or sparse visualizations. This is because DG Explainer accounts for temporal dynamics.
- **Sensitivity to threshold:** As threshold $\tau$ increases from 0.5 to 0.9, fidelity naturally decreases (fewer nodes retained), but DG Explainer maintains the highest fidelity across the entire range, indicating stable explanations.

### Limitations & Future Work

- On some datasets, slightly inferior to simpler methods in stability
- Future: adapt to transformers and attention mechanisms, dynamic heterogeneous graphs

---

## Part 2. DI Explainer (Sparse Attention Method)

**Speaker:** Bekoev Maksim

### Status

The audio recording for this presentation is **extremely garbled** and largely unintelligible. Only scattered fragments could be recovered:
- The method is based on **sparse attention** for interpretation
- Uses a **peer attention backbone**
- Mentions working with timestamps and snapshots
- Evaluation metrics discussed include fidelity, cross-entropy characteristics
- The method aims to provide attention-based interpretability

No reliable summary of the problem statement, method details, experiments, or results can be reconstructed from the recording.

### Discussion Highlights (Ilia Karpov)

- Mentioned connection to **continuous time modeling** and classical series
- Discussed whether the approach can be extended to **continuous time**
- Referenced a connection repository for the code

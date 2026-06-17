# Mentor's Seminar 13: TGB SEG Benchmark – Challenging Temporal GNNs

**Date:** Wednesday, May 27, 2026  
**Recording:** [Read.ai](https://app.read.ai/analytics/meetings/01KSMYXE1FWEWFYDP30G6JZQ0S?utm_source=Share_CopyLink)  

---

**Speaker:** Alexander Standrik

---
### Overview
Alexander presented the paper *"TGB SEG: Challenging Temporal GNNs with Complex Sequential Dynamics"*, which investigates why state‑of‑the‑art temporal GNNs fail on real‑world recommendation data despite performing well on traditional benchmarks like Wikipedia and Reddit.
**Key findings:**
- The authors observe that existing benchmarks have an unrealistically high **repeat ratio** (repeated edges between the same nodes). Models exploit this by memorising historical interactions rather than learning generalisable patterns.
- When evaluated on unseen edges, performance drops dramatically, confirming that the models rely on memorisation.
- The paper introduces a new **benchmark suite** with low repetition, power‑law degree distribution, and sparse interactions – more representative of real recommendation systems.
**Experimental results:**
- Eight temporal GNN baselines were evaluated on the new benchmark.
- No single model performs best across all datasets, highlighting the need for better architectures.
- Some models are computationally expensive, revealing a clear trade‑off between cost and effectiveness.
---
### Discussion Highlights
**Negative sampling strategy:**
- The benchmark uses random negative samples (garbage edges) for evaluation. The audience questioned whether random negatives are realistic – in recommendation, hard negatives (close but not clicked) would be more meaningful.
- Yuri Kondratov pointed out that purely random negatives are a weak baseline and that smarter strategies (e.g., Pinterest's hard negatives) are common in production systems.
**Over‑squashing vs. architecture:**
- Ilia Karpov raised the possibility that the problem is not the inability of GNNs to model dynamics, but rather **over‑squashing** of information in node memory. The paper does not ablate this, so the root cause remains unclear.
- The continuous temporal models share the same memory design, which may be the real bottleneck.
**Repository and reproducibility:**
- Alexander noted that the TGB benchmark repository has missing pages and a non‑functional leaderboard, raising concerns about code quality and reproducibility.
- The data loader seems to work, but the overall platform feels unfinished.
**Why repetition is considered a problem:**
- The authors distinguish between *same user buying the same item multiple times* (repetition) and *different users following the same pattern* (valuable signal). The new benchmark reduces the former while preserving the latter, making evaluation more rigorous.
---

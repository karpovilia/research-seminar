# Mentor's Seminar 15
**Date:** Wednesday, Jun 3, 2026  
**Recording:** [Read.ai](https://app.read.ai/analytics/meetings/01KT6ZE0NNC4Y4S4TMX0GARWYR?utm_source=Share_CopyLink)  

---

**Speaker:** Rustam Shangareev

---
### Overview
Rustam presented the paper *"SWE Reasoning: Dynamic Switching Between Discrete and Latent Sampling for Efficient Reasoning"* (submitted to ICLR 2026). The work introduces a **training‑free** decoding strategy that improves both accuracy and token efficiency in reasoning models.
**Core idea:**
- Standard reasoning models generate a long chain‑of‑thought (CoT) token by token using discrete sampling. This discards information (probabilities of other tokens) at each step.
- **Soft thinking** (proposed in 2025) replaces discrete tokens with weighted sums of token embeddings, allowing the model to "think" in continuous space. However, this leads to drift and noise over long chains.
- SWE Reasoning combines both approaches:
  1. **Dynamic switching** – the model monitors its **entropy** (confidence). When entropy is high (uncertain), it switches from discrete to latent (soft) sampling to explore alternative reasoning paths. When confident, it switches back.
  2. **Switch count control** – limits the total number of switches and softly terminates reasoning when the budget is exhausted, preventing over‑thinking and improving token efficiency.
**Results (as reported by authors):**
- Accuracy improvements of **1–5%** across math, coding, and general knowledge benchmarks.
- Token efficiency gains of **over 50%** (accuracy per generated token) on several datasets.
- Latent mode is used only in **0.3% of steps**, raising questions about its actual contribution.
---
### Critical Discussion
**Statistical significance:**
- Rustam performed a two‑sided z‑test on all reported comparisons – only 1 out of 28 turned out statistically significant. The authors did not address this, which is common in the field but still a concern.
**Unfair comparison under budget constraints:**
- For standard CoT, the authors simply truncated the reasoning at a fixed token limit. For SWE Reasoning, they used switch count control to gracefully terminate and allow a final answer. This gives SWE an unfair advantage in the token efficiency metric.
- A fair comparison would give CoT a termination token as well.
**Dead code and unimplemented features:**
- The paper describes two termination modes (convergence and termination), but the code only implements one. The unreachable termination path suggests the repository is incomplete.
**Why entropy?**
- Entropy is a crude measure of uncertainty. Other metrics (perplexity, attention‑based scores, or task‑specific difficulty) could be more informative for switching.
**Latent mode's limited use:**
- Since latent steps account for only 0.3% of all steps, the main benefit likely comes from the switch count control (which prevents over‑thinking), not from latent thinking itself. The paper's own ablation studies are not detailed enough to separate these effects.
**Reproducibility:**
- Rustam ran experiments on free GPUs and managed to reproduce results on GSM8K (improvement) but saw a drop on MATH500. The experiments are expensive (20 hours for 100 examples), making extensive validation difficult.
---
### Suggested Improvements
- **Adaptive window size** – instead of a fixed 512‑token window before switching, allow dynamic adjustment per task.
- **Fine‑tuning with the switching mechanism** – train models to better exploit latent thinking, reducing noise and improving quality.
- **Fair comparison with CoT** – give CoT a proper termination signal and compare under equal conditions.
- **Hyperparameter tuning** – the paper likely cherry‑picked β and window size; a more thorough ablation would strengthen the results.
---
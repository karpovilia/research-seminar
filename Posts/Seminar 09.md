# Mentor's Seminar 09

**Date:** Wednesday, May 13, 2026
**Recording:** [Read.ai](https://app.read.ai/analytics/meetings/01KRGXEVGRD2D2BQA9K9RS7BYK?utm_source=Share_CopyLink)

---

## Part 1: DIAL -- Aligning Large Language Models with Domain Adaptation (ICML 2025)

**Speaker:** Nick Evdokimov

A paper applying domain adaptation techniques from computer vision to LLM alignment, demonstrating that distribution alignment via Wasserstein distance and gradient reversal enables reward models trained on labeled source data to work on unlabeled target domains.

### Problem and motivation

- **Preference data is expensive and scarce** -- aligning models reliably requires thousands of labeled pairs, which cost significant time and money to produce.
- **Four practical pain points:**
  1. *Low-resource languages* -- ample English data exists, but Thai, Korean, and other languages have almost none.
  2. *Distribution mismatch* -- models trained on clean text fail on real user input with typos and slang.
  3. *Few-shot settings* -- only 10 labeled examples available, with thousands unlabeled.
  4. *Annotation cost* -- annotating full essays is expensive, but short fragments on the same topic are cheap.

### Core idea

- **Domain adaptation via distribution alignment:** instead of training on source data and hoping for generalization, explicitly force the model to build representations where source and target domains are indistinguishable in embedding space.
- If a Korean response becomes indistinguishable from an English one, the reward head trained on English automatically works on Korean.
- The task is **not** teaching the model a new task -- it is erasing the domain difference between two distributions.

### DIAL architecture

Two components on top of a base LLM (Gamma 2B with LoRA):

1. **Reward head** -- standard linear head trained on labeled source examples (normal reward model training).
2. **Critic** -- small network (two MLP layers with GELU) that classifies embeddings as source or target. Connected to the base model via a **gradient reversal layer**: on the forward pass it does nothing; on the backward pass it flips the gradient sign. The critic tries to distinguish domains, while the base model tries to fool it.

### Training objective

The full loss combines two terms:
- **Wasserstein distance** (with minus sign, because gradient reversal flips it) -- the critic maximizes the distance between source and target mean embeddings; the base model minimizes it.
- **Task loss on source data** (weight lambda = 0.01) -- standard reward model loss to maintain prediction quality.
- A **gradient penalty** enforces 1-Lipschitz constraint on the critic, preventing sharp changes that would break the Wasserstein formulation.

### Experiments and results

**Four scenarios (different types of domain gap):**

1. **Cross-lingual (Stanford Human Preferences):** English-trained model applied to Korean, Thai, Chinese. +5% average accuracy gain over zero-shot baseline. Slight regression on Chinese (baseline already strong due to clean semantic translation of child-level explanations).

2. **Style mismatch:** model trained on formal (LLM-rewritten) Reddit text, tested on noisy real Reddit posts. Without DIAL, accuracy drops from 0.70 to 0.67. With DIAL, it climbs to 0.73, closing the gap to the upper bound. Practical implication: train on clean annotation data, still handle messy production input.

3. **Few-shot generalization (10 labeled + many unlabeled):** training on 10 examples alone yields poor results (score near 1, far from upper bound). With DIAL, more than half the gap to full-data performance is closed purely from unlabeled data. PCA visualization confirms: DIAL clusters unlabeled points tightly around the few-shot examples, while the baseline scatters them.

4. **Easy-to-hard (fragments to essays):** annotating short fragments is cheap, annotating essays is expensive. Transfer from fragment-level to essay-level scoring. Pearson correlation improved from 0.502 to 0.571.

### Limitations and critique

- **Only Gamma 2B tested** -- a relatively small model; unclear whether results scale to production-grade models (7B, 70B+).
- **No ablation on labeled data count** -- only 10 vs. full dataset tested, no intermediate values (20, 30, 50). The mentor called this a methodological gap.
- **Chinese regression** unexplained -- when zero-shot baseline is already strong, DIAL provides no benefit, but the authors don't analyze why.
- **Only LoRA tested** -- no full fine-tuning experiments.
- **No comparison with other domain adaptation methods** beyond the ones cited as inspiration.

### Discussion highlights

- **Why 10 and not more?** The mentor questioned the lack of a labeled-data sweep. The speaker acknowledged this was a weak point and suggested results would likely improve with more labeled data but the DIAL gap would shrink.
- **Speaker's own experiments:** tested on Wikipedia translations of low-resource languages and confirmed the method reduces the embedding gap between languages, enabling the model to "think" in the target language rather than just translate.
- **Practical value:** DIAL can adapt an LLM to a new safety policy or domain with literally a dozen labeled cases, as long as unlabeled data is abundant. The labeled examples can even be generated by the LLM itself.
- **Super-alignment connection:** the style mismatch scenario is relevant to super-alignment -- when human annotators are noisy and models are powerful, the same logic applies in reverse.

---

## Part 2: Efficient Knowledge Injection in LLMs via Self-Distillation (PromptDistillation)

**Speaker:** Читранш

A paper proposing PromptDistillation (PD), a self-distillation method that injects new knowledge into LLMs without retraining from scratch, achieving performance close to RAG while being 3-10x more data-efficient than SFT.

### Problem and motivation

- **Knowledge becomes outdated quickly** -- models trained on data from years ago cannot answer questions about recent events.
- **Full retraining is expensive** -- rebuilding a model from scratch for every knowledge update is impractical.
- **Existing approaches have trade-offs:**
  - *SFT (Supervised Fine-Tuning)* -- effective but causes overfitting and mismatch between expert and student distributions.
  - *RAG (Retrieval-Augmented Generation)* -- industry standard, but requires context at inference time (longer prompts, higher latency, dependency on retrieval quality).

### Core idea: PromptDistillation

- **Teacher-student setup with the same base model** (self-distillation):
  - **Teacher** sees: knowledge document C + question Q + answer A.
  - **Student** sees: question Q + answer A only (no document).
- The student learns to match the teacher's **probability distribution over tokens** (soft targets via KL divergence), not the hard labels used in SFT.
- Intuition: the teacher reads from a textbook and explains; the student cannot see the textbook but learns to give nearly correct answers by imitating the teacher's reasoning.

### Key differences from SFT

| Aspect | SFT | PromptDistillation |
|--------|-----|-------------------|
| Training target | Hard labels (exact answer) | Soft probabilities (distribution) |
| Sampling temperature | Low (accurate) | High (creative, noisy) |
| Overfitting risk | High | Low (soft targets generalize better) |
| Data efficiency | Lower | 3-10x higher |

### Data generation pipeline

1. **High-temperature sampling** generates diverse questions and answers from the knowledge document. High temperature introduces noise, but this is acceptable because answers feed the teacher, not the student directly.
2. **KL distillation** -- the teacher produces token probabilities given C+Q+A; the student produces probabilities given only Q+A; the KL divergence loss minimizes the gap.
3. **Regularization** with 2-3 general datasets prevents catastrophic forgetting of pre-existing capabilities.

### Results

- **Closed-book accuracy:** PD achieves ~86% on AmazonQA, compared to lower SFT scores and ~87% for RAG. PD is within 1% of RAG without any context access at inference.
- **Data efficiency:** PD with only 20 questions matches SFT trained on ~200 questions. 3-10x more efficient at every LoRA rank tested (512, 768, 1024).
- **Multi-hop generalization:** PD was trained only on single-hop questions but generalizes well to HotpotQA (multi-hop). This confirms the model is not memorizing patterns but learning genuine reasoning.
- **PD + RAG:** combining both approaches achieves the **best overall performance**, surpassing RAG alone.
- **Catastrophic forgetting:** PD preserves general capabilities better than SFT, performing close to the original instruct model on general benchmarks while retaining injected knowledge.

### Larger teacher paradox

- Counter-intuitive finding: **larger teacher models do not always produce better students**.
- As the teacher grows, style mismatch, vocabulary mismatch, and capability mismatch increase, creating confusion rather than clarity.
- A larger teacher may spend more time "grasping" complex patterns, reducing the student's ability to react efficiently.

### Limitations and critique

- **No reproduction attempted** by the speaker; no GitHub link mentioned.
- **High LoRA ranks only** (512-1024) -- unclear if PD works with lower ranks common in resource-constrained settings.
- **No online/streaming updates** -- continuous distillation still risks catastrophic forgetting; knowledge injection is a batch process.
- **Does not surpass RAG alone** -- the main practical competitor (RAG) still outperforms PD. Only PD+RAG beats RAG.
- **Limited knowledge verification:** no mechanism to ensure the student actually learned specific facts (e.g., "Trump is president" vs. "Biden is president") rather than just approximating the teacher's distribution.

### Discussion highlights

- **When to use RAG over PD?** RAG remains better for very large models and when real-time knowledge access is critical. PD is preferable when inference latency, cost, or offline deployment are constraints.
- **Online training:** the mentor questioned why the paper claims no online/streaming updates are possible. In principle, mini-batch or single-sample updates should work. The speaker attributed the limitation to catastrophic forgetting under continuous distillation.
- **Controlling knowledge injection:** the mentor proposed a "soft shift" approach -- instead of forcing the student to fully adopt the teacher's distribution, gently shift it only where new information is needed, while preserving the original distribution elsewhere. This could prevent overwriting of existing knowledge.
- **Distilling cloud models:** the mentor raised the idea of extracting probability distributions from cloud APIs (Claude, GPT) to distill knowledge into local models. The challenge is that cloud models have different tokenizers and internal logic, which may break the distribution alignment.

---

## Part 3: Instruction Tuning for Large Language Models: A Survey

**Speaker:** Азамат Бейсеков

A comprehensive survey paper mapping the entire field of instruction tuning -- from data collection strategies to model families, evaluation benchmarks, and multimodal extensions.

### Problem and motivation

- **Fundamental mismatch:** base LLMs are pre-trained to predict the next token, but users want models to follow instructions clearly, safely, and in the correct format.
- **Rapid field growth:** the instruction tuning landscape is expanding so fast that a structured overview is needed to avoid getting lost.

### Instruction tuning pipeline

Two-step process:
1. **Build an instruction dataset** -- each example has an instruction, optional input, and desired output. Data sources: human-crafted, template-based (existing datasets), synthetic (stronger LLMs generate examples), or self-improvement (model expands its own data from a seed set).
2. **Fine-tune** the pre-trained LLM on instruction-output pairs via supervised learning.

### Taxonomy of instruction data

1. **Human-crafted** -- smaller but cleaner. Example: human-written task descriptions.
2. **Synthetic via distillation** -- strong models (GPT-4) generate examples for smaller models. Cheaper and scalable. Example: Alpaca (52K examples from GPT-4).
3. **Self-improvement** -- model starts from a small seed set and iteratively expands its own training data.

**Key insight:** data design is one of the most important parts of instruction tuning. The choice of data source significantly affects model behavior.

### Representative datasets and models

- **Datasets:** Supernatural Instructions (broad task coverage), Dolly 15K (open-source), Alpaca (synthetic, famous), LIMA (only 1,000 carefully selected samples -- quality over quantity).
- **Models:** Flan-T5 (open-family), Vicuna (popular open model), OpenChat, Perplexity.

### Multimodal and domain-specific extension

- Instruction tuning is not limited to chatbots -- the survey covers multimodal tuning (images, speech, video) and domain-specific applications (healthcare, finance, science).
- Efficient tuning methods (LoRA, QLoRA) are reviewed for compute-constrained settings.

### Evaluation landscape

- **Close-ended benchmarks:** MMLU, BBH (knowledge and reasoning).
- **Open-ended chat evaluation:** AlpacaEval, MT-Bench, Chatbot Arena (human preference).
- The survey synthesizes evaluation practices across many prior works rather than proposing new benchmarks.

### Limitations and critique

- **Some parts remain high-level** -- the breadth of coverage means depth is sacrificed in places.
- **Results not directly comparable** -- different papers use different models, datasets, and evaluation settings.
- **Open and closed models mixed** -- comparing proprietary (GPT-4) and open (Vicuna) models in the same framework can be misleading.
- **No benchmark matrix** -- a compact comparison table across models, datasets, and metrics would improve usability.
- **No decision guide** -- practitioners get a taxonomy but no clear recommendation on which strategy to choose for a given use case.

### Discussion highlights

- The speaker correctly identified the survey's value as **clarity and organization**, not a new benchmark score.
- The mentor had left due to time constraints before this presentation, so no mentor feedback was given.
- Three takeaways: (1) instruction tuning is simple but practical, (2) data choice matters enormously, (3) instruction tuning is essential but only one part of the larger alignment story (alongside RLHF, constitutional AI, etc.).

---

## Announcements

- The seminar ran over time; the third speaker presented after the mentor left.
- TIDFormer speaker (Azamat) was initially absent; his recording was to be sent separately, but he arrived and presented the survey in his place.

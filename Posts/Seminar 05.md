# Mentor's Seminar 05

**Date:** Thursday, Apr 23, 2026
**Recording:** [Read.ai](https://app.read.ai/analytics/meetings/01KPWW008XKHM22D7P1F8EJ3GZ?utm_source=Share_CopyLink)
---

## Part 1: Sign.MT -- Real-Time Multilingual Sign Language Translation Application

**Speaker:** Arman Malkhasyan

A system demonstration paper (EMNLP 2024) presenting an open-source web application for real-time bidirectional translation between spoken and signed languages, with a modular pipeline that lets researchers plug in their own models.

### Problem and motivation

- **Sign language translation is a cross-modal problem** -- converting between text, audio, and video adds complexity in compute, UI design, file handling, and platform support.
- **The research landscape is fragmented** -- different groups focus on different languages and pipeline stages with no unified platform.
- **Building interactive demos is expensive** -- server-side deployment costs money, client-side implementations are complex, discouraging researchers from creating demos.
- Sign.MT addresses all three issues: it is a **comprehensive open-source application infrastructure** where researchers can plug in models without building a full-stack app.

### System architecture

Two modular translation pipelines:

**Spoken-to-Signed (released):**
1. Input -- text or audio (speech-to-text via browser-native recognition).
2. Language identification -- MediaPipe detects the input language (104 languages supported in the UI).
3. Normalization -- optional text cleanup (e.g., using ChatGPT for capitalization, punctuation, grammar).
4. Sentence splitting -- browser-native internationalized segmentation.
5. Translation -- TranslateLocally (client-side MT engine) translates each sentence into **SignWriting** notation.
6. Animation -- SignWriting output converted to a pose sequence.
7. Rendering -- three visualization options:
   - *Skeletal viewer* -- minimalistic raw pose display; useful for developers and low-power devices.
   - *3D avatar* -- neural model translates body positions to a rigged 3D character.
   - *HumanGAN* -- pix2pix image-to-image model generates photorealistic avatar videos; highest quality but most compute.

**Signed-to-Spoken (under maintenance, planned release by end of spring):**
1. Input -- uploaded video or live camera capture.
2. Pose estimation -- MediaPipe Holistic extracts full-body pose from each frame.
3. Segmentation -- model detects boundaries between distinct signs.
4. Transcription -- segmented signs transcribed into SignWriting.
5. Translation -- SignWriting translated to spoken language text.
6. Output -- translated text with optional text-to-speech.

### Key technologies

- **SignWriting** -- central notation system (developed by Valerie Sutton, 1990) serving as the intermediate representation between modalities; captures handshapes, movements, and facial expressions.
- **MediaPipe Holistic** -- Google's real-time full-body pose estimation; runs on-device for offline capability.
- **TranslateLocally** -- client-side MT engine for text-to-SignWriting translation without server dependency.
- **Pix2Pix** -- conditional adversarial network for photorealistic avatar generation.
- **Pose-appearance transfer model** -- removes identifying information from poses for consistent avatar appearance.

### User adoption

- Over **5,000 users** in the US, 1,200 in India, hundreds across Europe, Philippines, Germany, Canada, and Israel -- all organic, none from the developer's home country.
- GitHub: **470 stars**, sustained community interest.
- Google Search Console: 3,750 clicks (up from 1,000) and 106,000 impressions (up from 24,400) over six months, achieved with a **single maintainer** and no marketing.

### Limitations and critique

- **Signed-to-spoken pipeline is not yet available** -- no recognition demo could be shown; only the generation side is functional.
- **No quantitative evaluation** -- no statistical metrics comparing translation accuracy against real users or baselines; no comparison of different models at pipeline stages.
- **Word-by-word translation** -- Alina Lobanova (guest expert in sign language linguistics) tested the system and found that Russian sign language output lacks proper grammar. The translation appears to map spoken words to signs in the same order, which does not reflect actual sign language grammar (which has its own word order). This could be done algorithmically without any ML training.
- **Facial expressions are critical but poorly rendered** -- in sign languages, facial expressions distinguish homonymous signs and mark questions (e.g., raised brows for yes/no questions). The 3D avatar technology does not adequately capture these, and diffusion-based avatars produce artifacts (hands disappearing).
- **Sign language dialects** are not accounted for -- Russian sign language has regional variations; the system does not model these.
- **Unclear training data provenance** -- it is unknown whether the system was trained on natural sign language or on the word-by-word "school" variant.
- **Computation vs. quality trade-off** is not discussed -- common phrases can be cached, but for arbitrary input, the quality-cost relationship across rendering options is undocumented.

### Discussion highlights

- The system's main practical value may be as a **corpus generation tool** for training sign language recognizers, rather than as a standalone translation product.
- The question of **generation drift** -- does training on avatar-generated signs help or hurt recognition of real human signing? -- remains unanswered.
- The offline capability and client-side processing are significant practical advantages for accessibility applications.

---

## Part 2: EM-LLM -- Human-Inspired Episodic Memory for Infinite Context LLMs

**Speaker:** Sergey Perov

A paper (ICLR 2025) from Huawei and UCL proposing a memory system for LLMs inspired by human episodic memory, enabling practically linear-time processing of sequences up to 10 million tokens without fine-tuning.

### Problem and motivation

- Modern LLMs struggle with long texts (>32K tokens) due to three issues:
  1. *Position problem* -- models trained on short text get confused with much longer sequences.
  2. *Speed problem* -- softmax attention is quadratic in sequence length; processing every word against every other word becomes prohibitively slow.
  3. *Information loss* -- when the model tries to remember too much at once, important details get washed out.
- Existing solutions (RAG, Infinite-LLM) help partially but still struggle with very long sequences.

### Architecture: three components

#### 1. Surprise-based segmentation

- **Bayesian surprise** is defined as: if minus log probability of the next token exceeds a threshold (mean + variance over a sliding window), a segment boundary is placed.
- Intuition: humans remember event boundaries where prediction errors spike (e.g., switching from a cooking book to mechanics is more memorable than content within the cooking section).
- Initial boundary points are set where surprise exceeds the threshold.

#### 2. Graph-based boundary refinement

- A graph is built over tokens within each candidate segment: for each attention head, key vectors at tokens *i* and *j* are compared via dot product similarity.
- Two graph-theoretic metrics are used to improve boundaries:
  - *Modularity* -- measures how well tokens cluster into communities.
  - *Conductance* -- measures the quality of the cut between clusters.
- Boundaries are iteratively adjusted to maximize these metrics. Complexity is O(N * M) where M is chunk size, much closer to linear than quadratic.

#### 3. Two-stage retrieval

When the model needs to recall information from its episodic memory:
- **Stage 1 -- Similarity buffer:** the current query is used to find the *k* most similar events via KNN search (FAISS for large memories). Events enter a buffer that functions as a queue -- old events leave as new ones arrive, implementing a natural **forgetting mechanism**.
- **Stage 2 -- Contiguity buffer:** for each retrieved event, its immediate temporal neighbors (+/- 1 event) are also retrieved. This mimics the human tendency to recall things that happened close together in time.
- The final context consists of the initial 128 tokens (attention sink), local context, and the retrieved episodic memory events.

### Experiments and results

- **Benchmarks:** LongBench and InfinityBench (up to **10 million tokens**).
- **Baselines:** RAG, Infinite-LLM (SOTA at time of publication), full-context (brute-force softmax).
- **Models tested:** Mistral, LLaMA, Phi, and others.
- **Ablation variants:** S (surprise only), M (refinement only), S+M, S+C (surprise + contiguity), full model (S+M+C).

**Key results:**
- On LongBench, EM-LLM outperforms baselines by **4.5--9% on average**, with up to **~40% improvement on retrieval tasks** and ~30% on QA.
- On InfinityBench, results are mixed -- Infinite-LLM sometimes performs better, but EM-LLM wins on tasks where memory recall is the critical factor.
- The **contiguity buffer (C)** alone performs surprisingly well, almost matching the full model on some tasks.
- Compared to RAG and full-context, EM-LLM shows large improvements on few-shot and coding tasks; other tasks are very close.
- **Human validation:** event boundaries from three podcasts were compared against human annotations. EM-LLM's boundaries (measured by modularity, conductance, and sustained distance) closely matched human annotations, while Infinite-LLM's fixed segmentation performed worse than random segmentation.

### Reproducibility

- **Code is public** on GitHub; the model is on HuggingFace.
- **Data is public:** LongBench, InfinityBench, PG19, and podcast annotations.
- Hyperparameters are documented in the appendix.
- Hardware requirements are high but achievable via cloud rental (~$8); GitHub issues confirm the code works.

### Limitations and future directions

- **Flat event structure** -- human episodic memory is nested (e.g., "making dinner" contains "chopping vegetables"), but EM-LLM uses flat, non-hierarchical events.
- **Modality-limited** -- currently text-only; extending to vision, audio, and multimodal inputs is future work.
- **Weak human validation** -- only three podcasts were annotated; more diverse open-source data is needed.
- **Simplistic forgetting mechanism** -- human memory involves more complex consolidation, interference, and rehearsal processes.

### Discussion highlights

- **Are surprises the right way to find boundaries?** The mentor questioned whether surprise-based segmentation is meaningful. If the text reads "the capital of France is London," the surprise occurs at "London," but this is a poor boundary. Surprise may be an *anchor* for an episode rather than a *boundary* -- the analogy to human cognition may be flawed in this regard.
- **Sensitivity to the surprise threshold** -- the threshold (mean + variance over a window) is tunable, but its impact on final performance is unclear. The refinement step (modularity-based graph clustering) may be doing most of the heavy lifting, making the initial surprise-based boundaries less critical than claimed.
- **Applicability to monotonous formal documents** -- formal documents (legal, technical) have more subtle topic shifts. The adaptive sliding window may handle this, but the paper does not specifically evaluate on such texts.
- **Alternative segmentation strategies** -- paragraph-based, section-based, or discourse-graph-based splitting may perform equally well or better; no comparison with non-surprise-based splitting was provided.
- The broader insight: EM-LLM is essentially building **discourse graphs** of text and performing community detection on token similarity graphs -- a well-studied approach in NLP, repackaged with a cognitive science framing.

---


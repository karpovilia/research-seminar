# Mentor's Seminar 04

**Date:** Wednesday, Apr 22, 2026
**Recording:** [Read.ai](https://app.read.ai/analytics/meetings/01KPTMZMXKW9YPMYWP55KTZKPN?utm_source=Share_CopyLink)
---

## Part 1: LightRAG -- Simple and Fast Retrieval-Augmented Generation

**Speaker:** Keshav Gupta

A paper proposing a graph-enhanced RAG framework that builds a knowledge graph from documents and uses dual-level retrieval to answer both specific and broad questions efficiently, dramatically reducing cost compared to GraphRAG.

### Problem and motivation

- **Naive RAG treats documents as flat, isolated chunks** -- it retrieves the most similar chunks but misses connections between them. Multi-hop questions like "How does the rise of electric vehicles influence urban air quality and public transportation infrastructure?" produce fragmented, incomplete answers.
- **GraphRAG solves this but is expensive** -- it uses community-level traversal requiring hundreds of API calls and hundreds of thousands of tokens per query, plus full graph rebuild on every data update.

### Graph-based indexing (RPD pipeline)

Three steps applied to every text chunk:
1. **Extract (R)** -- an LLM reads the text and pulls out entities (people, places, dates) as nodes and relationships as edges.
2. **Profile (P)** -- for every entity and relationship, creates a key-value pair: a short key phrase for quick search and a richer paragraph summarizing the entity (like a mini encyclopedia entry).
3. **Deduplicate (D)** -- merges duplicate entities that appear under slightly different names across documents.

The result is one large connected knowledge graph.

### Dual-level retrieval

Two simultaneous searches triggered by a user query:
1. **Keyword extraction** -- the system extracts *local keywords* (specific entity names, precise terms) and *global keywords* (broader themes, concepts).
2. **Keyword matching** -- local keywords match specific entity nodes; global keywords match relationship edges and themes. One-hop neighbors are also grabbed for additional context.
3. **Example:** for "movie recommendation metrics," local keywords are precision, recall, F1 score; global keywords are evaluation metrics, movie recommendation systems. Both are searched simultaneously.

### Experiments and evaluation

- **Benchmark:** UltraDomain -- college textbooks across four subjects: agricultural, computer science, legal, and mixed. Legal is the hardest (94 documents, ~5 million tokens).
- **Baselines:** NaiveRAG, RQ-RAG (sub-question decomposition), HaDE (hypothetical document embeddings), GraphRAG (community-based traversal).
- **Judge:** GPT-4o Mini, pairwise comparison across four dimensions: comprehensiveness, diversity, empowerment, and overall.
- **Results:** LightRAG achieves up to **84.8% win rate** against NaiveRAG on the legal dataset. It consistently wins on diversity. Against GraphRAG, the gap is smaller (~52--55%) but still positive.

### Cost analysis

- GraphRAG uses **610,000+ tokens** and hundreds of API calls per retrieval query on the legal dataset.
- LightRAG uses **fewer than 100 tokens** and only **one API call** -- over **600x cheaper** per query.
- LightRAG supports **incremental updates** -- new data adds nodes and edges without rebuilding the entire graph; GraphRAG requires full reconstruction.

### Ablation study

- Removing high-level retrieval worsens performance on broad questions.
- Removing low-level retrieval worsens performance on specific questions.
- **Most surprising result:** removing original text chunks entirely and relying only on the graph barely changes performance and sometimes improves it -- the graph captures important information so well that raw text adds noise.

### Limitations and critique

- **Only one model evaluated** (GPT-4o Mini) -- unclear whether results generalize to other foundation models or if the strong baseline performance is due to the model's capability rather than the method.
- **Wrong entity/relationship extraction** propagates errors through the entire knowledge graph with no correction mechanism.
- **Deduplication ambiguity** -- words with multiple meanings (e.g., Mercury the planet vs. Mercury the chemical) may be incorrectly merged; the paper does not address entity disambiguation.
- **Same model as reasoner and judge** -- using GPT-4o Mini for both generation and evaluation introduces bias (longer answers may be favored).
- **No comparison with NaiveRAG on simpler questions** -- prior research suggests ~70--80% of queries do not need graph retrieval; the uplift from LightRAG may be concentrated only on complex multi-hop questions.

### Discussion highlights

- The dual-level (local + global) retrieval design is elegant and practically useful for splitting queries into specific vs. broad categories.
- The 600x cost reduction with competitive quality is a strong practical argument for adoption.
- The speaker partially reproduced the results using Claude on a small corpus, comparing NaiveRAG and LightRAG on the four evaluation dimensions; initial results showed a tie before API issues interrupted further testing.
- Open question: what is the trade-off between using a cheaper model with larger context (GraphRAG-style) vs. a better model with minimal context (LightRAG-style)?

---

## Part 2: ThoughtTerminator -- Benchmarking, Calibrating, and Mitigating Overthinking in Reasoning Models

**Speaker:** Aleksandr Shulgin

A paper studying the phenomenon of overthinking in reasoning models -- generating unnecessary tokens that do not improve accuracy -- and proposing a technique to mitigate it.

### Problem and motivation

- **Reasoning models tend to overthink:** they generate large amounts of unnecessary tokens, continue reasoning after the correct answer has already been found, and repeat the answer multiple times in the chain before finally returning it.
- While there is a general correlation between tokens spent and task accuracy, models are poorly calibrated -- they waste compute on easy problems and do not allocate proportionally more to hard ones.

### Definitions of overthinking

1. Generating unnecessary tokens that do not improve accuracy.
2. Continuing to generate tokens after the answer was already reached.
3. The number of times the model repeats the correct answer in its reasoning chain before returning it.

### Measuring task difficulty

- **Per-model difficulty:** solve question *q* with model *m* *n* times; the failure rate (inaccuracy rate) is the difficulty measure.
- **Generalized difficulty:** average the per-model difficulty across a class of reasoning and non-reasoning models (DeepSeek, Qwen, Gemma, LLaMA) to get a model-independent estimate.
- **Key finding:** there is a clear positive relationship between question difficulty and the number of tokens the model spends. However, **variance is considerable** -- the same question can be answered correctly with 1,000 tokens in one trial and 2,000 in another, indicating significant room for improvement.

### Calibration metrics

- **Local overthinking score:** mean difference between average tokens spent (on correct answers) and minimum tokens spent, across questions.
- **Global overthinking score:** mean difference between maximum tokens spent and minimum tokens spent.
- Reasoning models spend **thousands of tokens** where non-reasoning models spend **hundreds**, with corresponding overthinking scores that are much higher.

### DAmP500 benchmark

- A new dataset of extremely easy tasks across four domains: math (e.g., "17 + 5"), chit-chat, common knowledge (e.g., "capital of France"), programming (e.g., "nth Fibonacci number"), and simple translation.
- Created because existing benchmarks for reasoning evaluation consist almost entirely of hard tasks, providing no easy questions to study overthinking on.
- Evaluation: numeric comparison for math, test coverage for code, LLM-as-judge for translation.

### ThoughtTerminator technique

- Before reasoning begins, the model is asked to **estimate how many tokens it needs** to solve the task (e.g., 600 tokens).
- During reasoning, ThoughtTerminator periodically interrupts the token stream with messages like *"N tokens remaining, hurry up."*
- When the token budget runs out, the system inserts *"Time is up, answer now"* and forces the model to immediately produce its answer.
- **Early exit optimization:** at each interrupt, the system checks (via regex) whether the model has already produced a final answer in its output stream; if so, the process terminates immediately.

### Results

- **Overthinking scores drop drastically** -- local and global scores reduced by roughly 1.5--2x.
- **Accuracy is largely preserved** -- drops slightly on some datasets but occasionally rises; on average, accuracy stays approximately the same.
- Several methods for estimating the initial token budget were tested: asking the model itself, using a separate model zero-shot, using GPT, and training a dedicated difficulty estimator.

### Limitations and critique

- **Pre-reasoning difficulty estimation is unreliable** -- questions with similar wording can have vastly different difficulty (e.g., "define company default" vs. "what is the probability of this company defaulting?"). The model cannot know the true difficulty before it starts reasoning.
- **Abrupt termination may cut off productive thinking** -- the model may be mid-thought when the budget runs out, losing the final reasoning step needed to reach the correct answer.
- **Temperature effects on difficulty estimation** are not discussed -- higher temperatures would increase variance in both difficulty estimates and reasoning paths.
- **No incremental deadline strategy** -- the current approach is a hard cutoff; a softer approach (e.g., granting a small extra budget when time is nearly exhausted) could improve results on complex questions.

### Discussion highlights

- The mentor proposed that **difficulty should be tracked as a process parameter during reasoning**, not estimated beforehand -- monitoring complexity as the reasoning unfolds would allow dynamic budget allocation, similar to how software developers spend 80% of time on the last 20% of work.
- A suggested experiment: instead of abrupt termination, give the model a **small extra token budget** (e.g., 100 tokens) as a last chance to complete its reasoning. The hypothesis is that this gentle extension would improve accuracy on hard questions without significantly increasing cost -- analogous to human deadline extensions.
- The broader theme across both papers: **complexity is not quality** -- simpler, faster approaches (LightRAG vs. GraphRAG, ThoughtTerminator vs. unbounded reasoning) can achieve comparable or better results at dramatically lower cost.

---



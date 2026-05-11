# Mentor's Seminar 07


**Date:** Thursday, Apr 30, 2026
**Recording:** [Read.ai](https://app.read.ai/analytics/meetings/01KPWW008XKHM22D7P1F8EJ3GZ?utm_source=Share_CopyLink)
---

## Part 1: Complete Chess Games Enable LLM Become a Chess Master

**Speaker:** Александр Уминский

A paper (NAACL 2025) introducing ChessLLM, the first large language model fine-tuned to play complete chess games from opening to checkmate, achieving an Elo rating of 1788 against Stockfish through supervised fine-tuning on a 20B-token dataset.

### Problem and motivation

- **Prior LLMs cannot play complete chess games** -- ChessGPT can select a single move from a position but cannot sustain a full game.
- **Move legality is not guaranteed** -- general-purpose LLMs (LLaMA, GPT variants) frequently output illegal moves, making them useless in real gameplay.
- **Evaluation has been weak** -- static NLP test sets do not capture actual chess ability; real gameplay against rated opponents is needed.

### Key innovations

1. **FEN instead of PGN** -- Forsythe-Edwards Notation encodes only the current board state at fixed length, while Portable Game Notation grows with every move and becomes too long for transformer context windows. This makes the task tractable for LLMs.
2. **Long-round Stockfish self-play data** -- a second dataset generated from Stockfish self-play at search depth 50--200 (much harder, higher-quality positions). Adding this data gives **+350 Elo improvement** -- the paper's most important empirical finding.
3. **Standard SFT pipeline** -- FEN-best-move pairs fine-tuned on OpenLLaMA 3B using causal language modeling.

### Experimental setup

- **Main experiment:** 100 games against Stockfish at skill levels 0, 1, and 2.
- **Move legality:** up to 10 sampling attempts per move to find a legal one.
- **Static evaluation set:** 10,000 board positions (weighted toward middle game) to prevent forgetting.
- **Head-to-head matches:** against LLaMA-7B, RedPajama, ChessGPT-Base, and ChessGPT-Check.
- **Metrics:** Pass@1, legality rate, best move accuracy, Elo rating, win/draw/loss counts.

### Results

- **Elo 1788** against Stockfish L2 (corresponds to roughly 1570--1720 human Elo).
- Win rates: 61% at L0, 56% at L1, 30% at L2.
- **>91% win rate against ChessGPT** in head-to-head matches.
- **97% best move accuracy** on positions from Elo 1700--3000.
- Clearly dominant in the LLM-vs-LLM category.

### Limitations and critique

- **Weak baselines** -- Stockfish L2 is a low ceiling; testing against L5--L8 or human players would be more convincing.
- **Model scale** -- only a 3B model tested; scaling laws suggest 7B or 13B could reach 2000+ Elo.
- **No reinforcement learning** -- SFT-only approach; incorporating RLHF or MCTS self-play (AlphaZero-style) could substantially improve results.
- **No open release** -- no code, dataset, or model weights published; methodology is transparent but non-reproducible.
- **Black side only** -- evaluation only on the black side; a two-color test would be more reliable.
- **Generalization claim without evidence** -- authors claim the method extends to shogi and Go but provide no experiments.

### Discussion highlights

- The mentor questioned whether **10 sampling attempts to find a legal move** undermines the claim of chess understanding -- if the model makes illegal moves, it does not truly understand the rules.
- **Game phase separation** -- openings and endgames are largely pattern-based and deterministic; middlegame strategy requires genuine positional intuition. The paper does not differentiate between phases.
- **Chess as a reasoning benchmark:** the mentor proposed that chess could serve as a reasoning benchmark for LLMs -- if Elo correlates with reasoning benchmark scores, chess-playing ability could be a proxy for general reasoning capability. Alexander noted the authors referenced related work but no explicit correlation study exists.
- **Strategic prompting:** the mentor suggested that instead of generating from all possible moves, presenting the LLM with a few strategy-aligned candidate moves and asking it to choose could improve results by leveraging the model's reasoning rather than its move-generation capability.

---

## Part 2: Fine-Grained and Thematic Evaluation of LLMs in Social Deduction Games

**Speaker:** Michael S

A paper (IEEE 2025) proposing a fine-grained evaluation framework for LLMs playing social deduction games (Spyfall), moving beyond coarse win-rate metrics to event-level analysis of reasoning behaviors.

### Problem and motivation

- **Coarse-grained metrics miss event-level behavior** -- prior work evaluates LLMs in social deduction games using only win rate and survival time, ignoring what actually happens during the game.
- **Non-systematic qualitative analysis** -- error analyses in prior work lacked structured, reproducible methodologies.

### Experimental environment: Spy Game

- Based on Spyfall: 7 players (1 spy + 6 citizens), citizens share a hidden location with distinct roles.
- **Spy wins** by correctly guessing the location or surviving the vote.
- **Citizens win** by identifying and voting out the spy.
- **Game renamed to "Spy Game"** to prevent models from relying on pre-trained knowledge of Spyfall.
- 7 locations, 24 games per location per model = **168 total game logs**, each averaging 5.4 turns and 23.8 reasoning steps.

### Models tested

- **Spy roles:** GPT-4, GPT-3.5, Gemini Pro, LLaMA 2.
- **Strong citizens:** GPT-4. **Weak citizens:** LLaMA 2.

### Fine-grained metrics

Two evaluation dimensions:

1. **Subtext inference** (spy's ability to deduce the location):
   - *Information catching rate:* exploitation of explicit verbal leaks (weak citizens only, since strong citizens rarely leak).
   - *Information deduction rate:* synthesis of implicit clues into correct guesses (strong citizens).

2. **Deceptive control** (spy's ability to avoid suspicion):
   - *Vote rate:* how often the spy is voted out (lower = better).
   - *Vote entropy:* how effectively the spy disperses suspicion among citizens (higher = better).

**Key finding:** LLaMA 2 has the highest survival time (coarse metric) but 0% information deduction rate -- it simply plays passively and never guesses. This exposes the failure of coarse metrics to capture actual ability.

### Thematic analysis: four categories of reasoning failure

Conducted by 3 annotators with CS backgrounds, iterative coding over several months:

1. **Exposure** (citizen error) -- a citizen literally mentions the hidden location, leaking information to the spy.
2. **Memory distortion** -- the spy invents or distorts prior conversational evidence (analogous to hallucination).
3. **Dissociation** -- the spy misattributes the source of a statement, treating its own words as coming from other players. Likely caused by generic labels ("Player 1", "Player 2") making source tracking difficult.
4. **Character ambiguity** -- the spy forgets its own role, thinking it is a citizen. Two subtypes:
   - *Goal misunderstanding:* the spy tries to identify and eliminate the spy (itself).
   - *Team misunderstanding:* the spy genuinely believes it is a citizen.

### Results

- **GPT-4 is best** against strong citizens (highest deduction rate, lowest failure counts).
- **LLaMA 2 is worst** across all metrics and has the highest incidence of all four failure categories.
- Memory distortion and dissociation directly cause lower subtext inference scores.
- Character ambiguity directly causes worse deceptive control (higher vote rate).

### Limitations and critique

- **Spy role only** -- citizens' behavior is not analyzed, despite exposure being a citizen-side failure mode.
- **Single game (Spyfall) only** -- no Mafia, no Werewolf, limiting generalizability.
- **Outdated models** -- LLaMA 2 and GPT-3.5 are not SOTA; newer models may behave differently.
- **Only 7 locations** -- a small set that may not exercise diverse reasoning strategies.
- **No code, no GitHub** -- all models and hyperparameters are listed but human annotation makes reproduction difficult.
- **No cross-game meta-evolution** -- in human Spyfall, strategy evolves over repeated games; this was not studied.

### Discussion highlights

- **Meta-evolution:** a listener suggested studying how strategies evolve across games, analogous to how human Spyfall players develop meta-strategies over time. The speaker agreed this would be valuable future work.
- **Causality of failures:** the mentor asked whether dissociation and exposure are *caused* by the model sensing it is losing, or are independent failure modes. Tracking the "balance" of the game (analogous to chess advantage scores) could reveal whether reasoning errors correlate with perceived disadvantage.
- **Source tracking:** the mentor noted that the dissociation problem may be solvable by giving players unique names or identifiers instead of generic "Player 1, Player 2" labels, reducing source confusion.

---

## Part 3: SimUSER -- Simulating User Behavior with LLMs for Recommender System Evaluation

**Speaker:** Ajith Arumugam

A paper proposing SimUSER, a framework that uses LLM-powered AI agents to simulate real user behavior for evaluating recommender systems, bridging the gap between offline metrics and expensive A/B testing.

### Problem and motivation

- **Offline metrics (RMSE, NDCG) do not reflect real user behavior** -- they measure prediction accuracy but miss engagement, satisfaction, and real-world interaction patterns.
- **A/B testing is expensive and slow** -- requires large numbers of real users, significant time, and production infrastructure.
- **Gap:** no scalable, cost-effective way to evaluate how real users interact with recommender systems.

### SimUSER architecture: three components

1. **Persona matching:**
   - Analyzes user's historical interaction data.
   - Generates multiple candidate personas representing possible user profiles.
   - Selects the persona that best matches the user (high internal similarity) while remaining distinct from other users' personas (high external dissimilarity), using a scoring function.

2. **Knowledge graph memory:**
   - **Episodic memory:** text records of past interactions (what items the user liked/disliked).
   - **Structural memory:** a knowledge graph where nodes represent items and edges represent relationships (genre, shared features, actors, etc.).
   - Similarity between items computed via connection count in the graph, normalized by total connections.

3. **Brain model (preference, action, reflection):**
   - Evaluates user preference for recommended items.
   - Rates items and decides actions: watch, skip, or exit.
   - Considers satisfaction level after each interaction.
   - Performs **reflection** after each step to update memory, making the system adaptive over time.

### Results

- **~79% accuracy** in predicting user preferences (highest among compared approaches).
- **Lowest RMSE** -- predictions closest to actual user behavior, attributed to knowledge graph memory enabling better generalization.
- **Engagement score ~4.41/5** -- simulated behavior closely resembles real user engagement patterns.
- Authors report **55 proprietary A/B tests** showing correlation between SimUSER predictions and real-world outcomes.

### Limitations and critique

- **No code, no dataset, no GitHub** -- fully closed-source. Impossible to reproduce.
- **Proprietary A/B test data** -- 55 tests claimed but no details: how many users, statistical significance, test duration, or whether tests were identical or diverse. Cannot be validated.
- **Unclear target metric** -- the mentor repeatedly asked what specific business metric is optimized. "Engagement" is vaguely defined; its mapping to business outcomes (retention, revenue) is not established.
- **50+ events per user** is unrealistic for many real recommender systems that operate with cold-start users or sparse interaction histories.
- **Outdated models** -- no comparison with current SOTA LLMs.
- **No comparison with classical RS algorithms** -- it is unclear whether AI agents add value over well-tuned collaborative filtering or content-based methods.

### Discussion highlights

- The mentor criticized the vague "engagement" metric: "Can we just come to a user and ask, are you engaged or not?" The metric's computation and its practical relevance are unclear.
- **Proprietary data problem:** the only connection to real-world data is a set of 55 A/B tests that cannot be inspected. All other experiments are based on retrospective evaluation, which is a fundamentally different evaluation paradigm.
- **Cost efficiency argument** was the speaker's main defense: SimUSER reduces the need for expensive real-user A/B tests. The mentor acknowledged the cost argument but maintained that without validated metrics, the savings are meaningless.
- The mentor compared the approach to chess LLMs: if the target metric is not clearly defined and optimized, the system may appear to work without actually providing actionable value.

---

## Announcements

- The mentor noted that several papers in this session used **outdated models** and encouraged speakers to consider how SOTA models would affect the reported results.
- The seminar ran over time; questions for the final speaker were cut short.

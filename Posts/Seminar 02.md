# Mentor's Seminar 02

**Date:** Thursday, Apr 9, 2026
**Recording:** [Read.ai](https://app.read.ai/analytics/meetings/01KNRTFY3GH8APZ8YME2W0C83B?utm_source=Share_CopyLink)
---

## Part 1: LLM-Powered Collaborative Task Planning Framework (ICAPS 2025)

**Speaker:** Petr Shortahila

A paper proposing a system that allows domain experts to participate in collaborative planning by expressing constraints in natural language, without requiring knowledge of planning formalisms.

### Problem and motivation

- **Collaborative planning** -- multiple experts contribute insights to build structured plans for solving optimization problems, but participation requires knowledge of formal planning notations, creating a barrier for domain experts unfamiliar with these formalisms.
- **Core idea:** build a service that lets domain experts contribute their knowledge through natural language, with the system translating it into machine-executable planning constraints.

### Architecture

- The system receives a **problem description and its visualization**, then experts provide constraints in **natural language**.
- **Two-stage decomposition pipeline** -- the main contribution of the paper:
  1. LLM (SONNET-4) decomposes natural language constraints into simpler components, asking the expert for clarification to prevent misunderstandings.
  2. Decomposed constraints are translated into **PDDL3** (Planning Domain Definition Language), with a second round of expert feedback for corrections.
- A **verifier** checks syntactic correctness of generated PDDL3 constraints.
- Valid constraints are passed to the **ENHSP planner**, which produces a plan visualized via **PDC** (Planner Domain Construction tool) so experts can see the effect of their constraints visually.

### Example: XenoTravel

- Cities (rectangles), people (circles), planes (triangles); each person has a goal destination. People travel by plane, each flight consumes fuel. Goal: everyone reaches their destination with minimal fuel.
- Expert constraint: *"planes 2 and 3 have higher fuel consumption"* -- the model translates this to "use only plane 1," and the planner adjusts accordingly.

### Limitations and critique

- **Only state-based constraints** are supported (predicates), no action-based constraints (e.g., "plane 1 should fly only from city 1 to city 2") -- this is a fundamental limitation of PDDL3, not just the system.
- **No fine-tuning** of the LLM -- authors used off-the-shelf open-source models, which likely hurts constraint translation quality.
- **No comparison** with alternative approaches, no error analysis, no measurable evaluation.
- **Reproducibility is poor** -- no GitHub repo, no GUI, no code; only a two-page paper and an 8-minute demo video with a robotic voice.
- PDDL3 is structurally similar to **context-free grammars** -- actions lead to results described by production rules, analogous to CFG derivations.

### Discussion highlights

- The seminar discussed whether **zero-shot prompting is sufficient** or whether fine-tuning (e.g., LoRA) is necessary for reliable natural-language-to-PDDL conversion, drawing parallels with Text-to-SQL systems.
- Comparison with **BPMN** (Business Process Model and Notation) as an alternative formalism for expressing plans -- the question of which formalism is most powerful for planning tasks remains open.
- The paper was compared unfavorably to a SIGIR demo submission that was rejected despite providing a repository (albeit with a reproducibility score of 1/5).
- The idea of using LLMs to formalize planning tasks was considered promising as a **starting point for a course paper or graduate work**, especially with contemporary models (e.g., SONNET-5) instead of SONNET-4.

---

## Part 2: TAPAS -- Task-Adaptation and Planning Using Agents (2025)

**Speaker:** Danila

A multi-agent framework combining LLMs with symbolic planning to solve real-world tasks, demonstrating the ability to adapt to new domains and retain knowledge across tasks.

### Problem and motivation

- **LLMs alone** are good at text transformation but poor at making structured plans.
- **Symbolic planners alone** are good at planning but require precisely defined, passive domains -- they cannot generate domain descriptions themselves.
- **TAPAS** bridges the gap: LLMs build the domain model, symbolic planners solve it.

### Architecture

- **Domain modeling agent (LLM-powered):**
  - *Domain generator* -- converts natural-language task description into PDDL domain code (object types, predicates, actions).
  - *Initial state generator* -- produces initial state values from the text description.
  - *Goal state generator* -- determines the specific goal (e.g., "I want an apple from my bag" -> which apple, where it is).
  - Generators work **independently but with upward feedback** -- lower-level generators update the domain generator, iterating until a satisfiable plan model is reached.
  - **Memory mechanism:** short-term memory (within-task, standard GPT context) and **procedural (long-term) memory** -- computes similarity with previous tasks and reuses knowledge from analogous tasks to handle new, more complex problems.
  - **Self-reflection mechanism:** an internal critic assigns a quality score (0--1) to the model; if above a threshold, the process stops; otherwise, it iterates up to a limit.

- **Symbolic planning agent:**
  - *Solver* -- receives PDDL code from the LLM agent and generates a sequence of plan steps.
  - *Plan abstraction* -- translates formal plan steps back into natural language for user comprehension.
  - *Plan executor* -- executes the plan and checks whether the goal is satisfied; if not, feedback is sent back to improve the model.

### Experiments and results

- **Experiment 1:** 7 planning domains from the LLM-PlanBench, GPT-4 class model, temperature 0, 10 iteration limit. Average accuracy **>88%**, with strong performance across most domains.
- **Experiment 2 (ablation):** tested temperatures 0, 0.1, and 0.3 -- **0.1 is optimal**; accuracy drops at higher temperatures.
- **Experiment 3 (domain adaptation):** added new features to existing tasks (colors and sizes in Blocksworld, battery levels). Accuracy reached **70--100%**, demonstrating that long-term memory enables successful transfer to upgraded domains.
- **Experiment 4 (VirtualHome simulation):** simulated real-world household tasks (take food from fridge, eat, place on table). The agent also remembered a persistent instruction ("always close the fridge door") and applied it to new tasks without being reminded, confirming long-term memory utility.

### Limitations and critique

- **Quality score is opaque** -- no formula provided, only described as a 0--1 scale based on plan executability and information completeness.
- **Tasks are very simple** -- Blocksworld, Tireworld, VirtualHome; the most complex example is changing tires, which is far simpler than a real garage procedure.
- **No discussion of ontological contradictions** -- as the knowledge base grows, conflicting facts may accumulate with no resolution mechanism.
- **No low-level planning** -- the framework operates only at high-level abstraction (e.g., "open fridge," "take food"); real-world robotics requires fine-grained motor control (step forward, raise hand, grasp handle), which is not addressed.
- **Plan evaluation is text-based** -- similarity scoring compares text descriptions, but reasonable-sounding plans may fail during execution (the "Neocon law" problem).
- **Reproducibility concerns** -- the speaker did not attempt to run the code, believing it required special OpenAI research resources; the mentor noted that modern cloud GPUs ($10/day for H100) and coding assistants make reproduction feasible.

### Discussion highlights

- **High-level vs. low-level planning:** should agents learn only abstract plans or also detailed motor sequences? The speaker proposed a hierarchical approach -- TAPAS generates high-level plans, while a separate "brain" inside the robot handles low-level execution (analogous to brain vs. spinal cord).
- **Domain adaptation and memory:** skills from low-level designs could be reused across domains, but high-level designs would require significant retraining. The mentor noted this distinction itself could be a research paper.
- **Software development analogy:** typical development workflows (design, backend, frontend, testing, deployment) are also planning problems with reusable "bricks" -- measuring plan quality when multiple valid plans exist remains an open question.
- The mentor emphasized that future speakers should **at least attempt to run the code** from papers they present, using available cloud tools and coding assistants.

---


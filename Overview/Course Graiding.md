# Course Grading

## Presentation

Presentations can be prepared **individually** or **in pairs**:

| Format | Number of papers |
|---|---|
| Individual | 1 paper |
| Pair | 3 or more papers (topic survey) |

### Presentation structure

| Section | Points | Guiding question |
|---|---|---|
| **Problem statement** | 1 | What problem do the authors solve? |
| **Key idea** | 1 | What is the novelty of the paper? |
| **Experiment** | 1 | How do the authors test their hypothesis? |
| **Results** | 1 | Did they beat SOTA? |
| **Contribution** | 1 | What is the main contribution? |
| **Reproducibility** | 2 | Can the experiments be reproduced? Is the code available? Does it run? |
| **Improvements** | 2 | What could be improved in the paper? |
| **Presentation quality** | 1 | Slide design and overall formatting |
| **Total** | **10** | |

## Grading Formula

The final grade consists of three components:

### 1. Seminar activity (questions)

- **1 point** per question asked during a seminar
- Maximum **3 activities** per course
- Total: up to **3 points**

### 2. Peer reviews

- **1 point** per review of another student's presentation
- Maximum **2 reviews** per course
- Total: up to **2 points**

### 3. Presentation (weight 7)

Each presentation is evaluated by **3 reviewers**: the mentor (L) and two peer reviewers (M, N).

The presentation score is a **weighted average** with the following weights:

| Reviewer | Weight |
|---|---|
| Mentor (L) | 2 |
| Peer reviewer 1 (M) | 1 |
| Peer reviewer 2 (N) | 1 |

**Presentation score formula:**

$$
S = \begin{cases}
0, & \text{if } w_L + w_M + w_N = 0 \\[6pt]
\dfrac{L \cdot w_L + M \cdot w_M + N \cdot w_N}{w_L + w_M + w_N}, & \text{otherwise}
\end{cases}
$$

where

$$
w_i = \begin{cases}
2, & i = L \text{ (mentor), if grade is given} \\
1, & i \in \{M, N\} \text{ (peer reviewer), if grade is given} \\
0, & \text{if grade is missing}
\end{cases}
$$

### Final formula

$$
\boxed{\;\text{Total} = \underbrace{\min(Q,\, 3)}_{\text{questions}} + \underbrace{\min(R,\, 2)}_{\text{reviews}} + \underbrace{S \times 7}_{\text{presentation}}\;}
$$

where $Q$ — number of questions asked, $R$ — number of peer reviews written, $S$ — weighted average presentation score.

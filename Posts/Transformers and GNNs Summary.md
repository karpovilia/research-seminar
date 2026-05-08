# Mentor's Seminar 08

**Date:** Tuesday, May 6, 2026

---

## Part 1: Transformers are Graph Neural Networks -- On the Equivalence Between Multi-Head Attention and Message Passing

**Speaker:** Лев Вартанов

A theoretical paper establishing a formal equivalence between multi-head attention in transformers and message passing in graph attention networks (GATs), arguing that transformers are GNNs that won the hardware lottery due to highly optimized dense matrix operations on modern GPUs.

### Background: from RNNs to transformers

- **Recurrent Neural Networks** process tokens sequentially (autoregressive, left to right), producing latent embeddings for each word. Their sequential nature limits long-range dependency capture.
- **Attention mechanism** allows models to focus on different parts of the input simultaneously, replacing sequential processing with parallel weighted aggregation.
- **Transformer architecture:**
  - Input tokens S mapped through three learnable projections: WQ (queries), WK (keys), WV (values).
  - Updated embedding for each token = weighted sum of all value vectors, with weights from softmax(Q_i * K_j^T).
  - **Multi-head attention** computes multiple attention maps in parallel, each with its own Q/K/V projections. Outputs are concatenated and projected back via a linear transformation (W_O).
  - Residual connection: output = previous embedding + attention output, followed by layer normalization.

### Graph neural networks

- Graphs model interconnected systems (social networks, molecular structures) as sets of nodes connected by edges with no fixed ordering.
- **GNNs operate via message passing:**
  1. For each node, construct messages from neighboring nodes representing pairwise relationships.
  2. Aggregate messages using a permutation-invariant operator (sum, min, max).
  3. Update node representation with the aggregated result and a learnable transformation.
- Stacking multiple message-passing layers propagates information beyond immediate neighbors.
- **Graph Attention Networks (GATs)** use attention mechanisms to weight neighbors by importance during aggregation, with multi-head variants for capturing different relationship aspects.

### Core equivalence argument

- Multi-head attention in transformers is formally identical to message passing in GATs -- the attention formula is the same, except:
  - **Transformers** operate on a **fully connected graph** (every token attends to every token) -- global attention.
  - **GATs** operate on the **graph's adjacency structure** -- local attention, only aggregating from actual neighbors.
- This means **transformers are GNNs on complete graphs**, and can learn to capture both local and global context.
- This equivalence has inspired **graph transformers** -- a new class of models that inject graph structure information (e.g., Laplacian positional encodings) into transformer attention to combine local message passing with global multi-head attention.

### Hardware lottery

- Despite theoretical equivalence, there is a critical practical difference:
  - **Transformers** use dense matrix multiplication, which is **highly optimized for modern GPUs** (CUDA, Tensor Cores).
  - **GNNs** use sparse data structures that are **poorly optimized** for current GPU architectures.
  - GNNs only outperform on very sparse or billion-scale graphs.
- **Conclusion:** transformers are GNNs that won the hardware lottery. The performance gap is largely due to GPU optimization, not architectural superiority.

### Speaker's independent experiments

Lev attempted to verify the theoretical equivalence empirically:

1. **CORA benchmark (node classification on citation graph):**
   - 6-layer GAT vs. 6-layer transformer with Laplacian positional encoding.
   - **Result:** Transformer performed very poorly, nearly at random classifier level, with high variance across seeds. GAT performed much better.
   - Possible explanations: the positional encoding did not adequately inject graph structure; or the theoretical equivalence does not translate to practical optimization landscapes.

2. **AG News (text classification, no graph structure):**
   - GAT (fully connected graph) vs. transformer on text data.
   - **Result:** both models performed better than random, but GAT showed higher accuracy and significantly lower variance than the transformer.
   - This was unexpected -- without graph structure, the two architectures should be more equivalent.

### Limitations and critique

- **Purely theoretical review** -- no empirical experiments in the original paper to validate whether the mathematical equivalence translates to practical performance.
- **Broad conclusions** -- the author claims equivalence between transformers and GNNs, but only formally shows equivalence between transformers and GATs (attention-based GNNs). The connection to non-attention GNNs (GCN, GraphSAGE) is "far more tenuous" and lacks theoretical support.
- **Graph transformers undercovered** -- the paper barely addresses architectures that explicitly combine GNN and transformer strengths, which is arguably the most practically relevant direction.
- **Reproducibility concerns** -- the speaker noted that graph embeddings from different implementations are often incompatible, even when theoretically equivalent. This implementation-level divergence may undermine the paper's clean mathematical framing.

### Discussion highlights

- **Attention matrix comparison:** the mentor suggested a direct test -- if GAT and transformer implementations produce the same attention matrices for the same input, the equivalence is validated at the implementation level (not just mathematically). Different frameworks make this non-trivial.
- **Transfer learning thought experiment:** if attention matrices are truly equivalent, one could train weights as a transformer and transfer them to a GAT for inference. The GAT's adjacency mask would zero out non-neighbor entries, and if accuracy does not drop, the equivalence holds practically. The mentor called it "a funny experiment" but acknowledged the QKV differences between the two architectures make this challenging.
- **The "everything is a transformer" genre:** the mentor noted this is a known category of theoretical papers (Bayesian transformers, etc.), and this author has written similar works. While not groundbreaking, the explicit connection to the hardware lottery provides practical insight.
- **Signal propagation view:** both architectures model signal propagation through layers; the key difference is how the signal pathway is defined (dense vs. sparse). Understanding this connection is valuable even if practical transfer is limited.

---

## Announcements

- The seminar scheduled for Thursday, May 7 was cancelled due to student vacations. Future time slots will be rescheduled.
- The mentor evaluated the speaker's performance immediately after the session.

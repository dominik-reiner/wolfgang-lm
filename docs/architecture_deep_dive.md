# Wolfgang-LM: Architecture Technical Specification

## 1. System Specifications
*   **Model Type**: Decoder-Only Transformer (Llama-style)
*   **Parameters**: ~75 Million
*   **Context Window ($T$)**: 512 Tokens
*   **Model Dimension ($d_{model}$)**: 640
*   **Batch Size ($B$)**: Flexible (Traced here as $B=1$)

### Phase 1: Embedding (The Input)
| Step | Shape In | Operation | Shape Out | Mathematical Definition |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | `[512]` | Lookup | `[512]` | $x_t \in \{0, ..., V-1\}$ (Indices) |
| **Embed** | `[512]` | `Embed[x] ` | `[512, 640]` | $E = W_{emb}[x]$, where $W_{emb} \in \mathbb{R}^{V \times d_{model}}$ |

### Phase 2: The Block Loop (Repeated 12 Times)
Input to Block $i$ is $x_i \in \mathbb{R}^{1 \times 512 \times 640}$.

#### Sub-Layer A: Attention (GQA)
| Step | Shape In | Operation | Shape Out | Rationale & Background |
| :--- | :--- | :--- | :--- | :--- |
| **Norm** | `[512, 640]` | `RMSNorm` | `[512, 640]` | **Zhang et al. (2019)**: Proved mean-centering is redundant. Only variance scaling matters for convergence. |
| **Project** | `[512, 640]` | $x \cdot W^T$ | $Q$: `[512, 640]`<br>$K$: `[512, 320]`<br>$V$: `[512, 320]` | **Weights**: $W_Q \in \mathbb{R}^{640 \times 640}$<br>$W_K \in \mathbb{R}^{320 \times 640}$<br>$W_V \in \mathbb{R}^{320 \times 640}$.<br>**Math**: $[512, 640] \cdot [640, 320] \rightarrow [512, 320]$. <br>**GQA (Ainslie et al., 2023)**: Projects to smaller rank. |
| **Split** | `[512, 640]` | `View` | $Q$: `[10, 512, 64]`<br>$K$: `[5, 512, 64]`<br>$V$: `[5, 512, 64]` | Logically separates subspaces. The columns of $W_Q, W_K, W_V$ are pre-trained to sort features into these specialized "bins". |
| **RoPE** | `[10, 512, 64]` | `Rotate` | $Q$: `[10, 512, 64]`<br>$K$: `[5, 512, 64]`<br>($V$ is unchanged) | **Su et al. (2021)**: Applies $x'_m = x_m e^{i m \theta}$ in complex plane to $Q$ and $K$. Encodes relative distance $m-n$ for the attention score. **$V$ carries payload and is NOT rotated.** |
| **Expand** | `[5, 512, 64]` | `Repeat` | $K, V$: `[10, 512, 64]` | **Implementation Detail**: Duplicates KV heads in memory ($2\times$) to match Q heads for matrix multiplication. |
| **Score** | `[10, 512, 64]` | $Q \cdot K^T$ | `[10, 512, 512]` | Raw Attention Scores. Mask ensures $Scores_{ij} = -\infty$ if $j > i$ (Masked Causal). |
| **Softmax** | `[10, 512, 512]` | `Softmax` | `[10, 512, 512]` | The Attention Map $A = \text{Softmax}(Scores)$. Converts scores to probabilities ($\sum=1$). |
| **Context** | `[10, 512, 512]`| $A \cdot V$ | `[10, 512, 64]` | Computes weighted sum of values. **Math**: $[512 \times 512] \cdot [512 \times 64] \rightarrow [512 \times 64]$. Each head produces a vector in $\mathbb{R}^{64}$. |
| **Merge** | `[10, 512, 64]` | `View` | `[512, 640]` | Concatenates heads. |
| **Output** | `[512, 640]` | $x \cdot W_O^T$ | `[512, 640]` | **Math**: $[512, 640] \cdot [640, 640] \rightarrow [512, 640]$.<br>**Weights**: $W_O \in \mathbb{R}^{640 \times 640}$.<br>**Low-Rank Interpretation**: Mixes the 10 heads back into the model manifold. |
| **Skip Connection** | `[512, 640]` | `Add` | `[512, 640]` | **Element-wise Addition**: $y = x + F(x)$. The block learns the **Residual** (the difference needed to improve $x$). If $F(x) \approx 0$, the layer does nothing (Identity). |

### Conceptual Note: "Low-Rank" View of Value + Output Projection (Interpretation)

> **Important:** Wolfgang-LM currently uses standard dense linear layers for $W_V$ and $W_O$ (no explicit low-rank factorization / LoRA modules). The "low-rank" language below is a **conceptual lens** that can help build intuition for why attention is structured as "project $\rightarrow$ mix $\rightarrow$ project".

#### The idea
Attention can be understood as moving information through a smaller **value feature space** (per head) and then mapping it back into the full model space.

For a single head ($h$), define:
*   Model dimension: $d = d_{model}$
*   Per-head value dimension: $d_v$ (in Wolfgang-LM, $d_v = d_{head} = 64$)

**(1) Value-down projection (the "V matrix" per head)**
We project token states ($x_t \in \mathbb{R}^{d}$) into a smaller per-head value space:
$$v_t^{(h)} = x_t W_{V,h}^T \in \mathbb{R}^{d_v}, \quad W_{V,h} \in \mathbb{R}^{d_v \times d}$$

**(2) Weighted sums happen in the smaller value space**
Given attention weights ($a_{ts}^{(h)}$), the head output in value space is:
$$o_t^{(h)} = \sum_{s \le t} a_{ts}^{(h)} v_s^{(h)} \in \mathbb{R}^{d_v}$$
This is the key computational pattern: the expensive token-mixing (weighted sums) is done over vectors of dimension $d_v$, not $d$.

**(3) Value-up projection (often "hidden" inside the output matrix)**
Each head's value-space output is mapped back into model space:
$$\Delta x_t^{(h)} = o_t^{(h)} W_{U,h}^T \in \mathbb{R}^{d}, \quad W_{U,h} \in \mathbb{R}^{d \times d_v}$$

#### Why this is called "low-rank" (conceptually)
The composition "down then up" forms a rank-bounded mapping: $W_{U,h}^T W_{V,h}^T$ has rank at most $d_v$. So each head's read/write pathway can be viewed as a **rank-($\le d_v$) route** from model space back to model space (with attention providing the data-dependent mixing across tokens).

> [!NOTE]
> In many architectures, $d_v = d_{head}$ (as in Wolfgang-LM), so this is mainly an **interpretation**. Some designs choose $d_v \ll d_{head}$ to *explicitly* reduce compute by doing the weighted sums in a stricter bottleneck space.

#### Why the "up projections" are bundled into a single $W_O$
After attention mixing, we concatenate all head outputs:
$$o_t = \mathrm{concat}(o_t^{(1)}, \dots, o_t^{(H)}) \in \mathbb{R}^{H d_v}$$
Instead of applying each $W_{U,h}$ and summing, implementations usually do one matrix multiply:
$$\mathrm{out}_t = o_t W_O^T$$
This is equivalent because $W_O^T$ can be seen as the vertical stacking of the per-head "up" matrices:
$$W_O^T = \begin{bmatrix} W_{U,1}^T \\ W_{U,2}^T \\ \vdots \\ W_{U,H}^T \end{bmatrix}$$
Therefore:
$$o_t W_O^T = \sum_{h=1}^{H} o_t^{(h)} W_{U,h}^T$$
which matches "apply each head's up-projection and add" exactly.

#### Sub-Layer B: MLP (SwiGLU)
| Step | Shape In | Operation | Shape Out | Rationale & Background |
| :--- | :--- | :--- | :--- | :--- |
| **Norm** | `[512, 640]` | `RMSNorm` | `[512, 640]` | Stabilizes input to MLP. |
| **Gate** | `[512, 640]` | $x \cdot W_{gate}^T$ | `[512, 1728]` | **Math**: $[512, 640] \cdot [640, 1728] \rightarrow [512, 1728]$.<br>**Weight**: $W_{gate} \in \mathbb{R}^{1728 \times 640}$ (Stored).<br>**Shazeer (2020)**: Creates a "gating" probability. |
| **Value** | `[512, 640]` | $x \cdot W_{val}^T$ | `[512, 1728]` | **Math**: $[512, 640] \cdot [640, 1728] \rightarrow [512, 1728]$.<br>**Weight**: $W_{val} \in \mathbb{R}^{1728 \times 640}$ (Stored).<br>The raw feature transformation. |
| **Act** | `[512, 1728]` | `Swish * Val` | `[512, 1728]` | **SwiGLU**. Operations:<br>1. $\text{Swish}(Gate) = Gate \otimes \text{Sigmoid}(Gate)$<br>2. $Result = \text{Swish}(Gate) \otimes Value$<br>Uses a $\frac{8}{3}$ scaling ratio ($1728 \approx \frac{8}{3} \times 640$) to maintain parameter parity. |
| **Down** | `[512, 1728]` | $x \cdot W_{down}^T$ | `[512, 640]` | **Math**: $[512, 1728] \cdot [1728, 640] \rightarrow [512, 640]$.<br>**Weight**: $W_{down} \in \mathbb{R}^{640 \times 1728}$.<br>Compresses back to model manifold. |
| **Skip Connection** | `[512, 640]` | `Add` | `[512, 640]` | **Element-wise Addition**: $y = x + F(x)$. Adds the MLP's non-linear feature transformation to the stream. |

### Phase 3: Classification
| Step | Shape In | Operation | Shape Out | Function |
| :--- | :--- | :--- | :--- | :--- |
| **Norm** | `[512, 640]` | `RMSNorm` | `[512, 640]` | Final stabilization. |
| **Logits** | `[512, 640]` | $x \cdot W_{head}^T$ | `[512, 32768]` | **Math**: $[512, 640] \cdot [640, 32768] \rightarrow [512, 32768]$.<br>**Weights**: $W_{head} \in \mathbb{R}^{32768 \times 640}$.<br>Projects from Model Space $\to$ Vocab Space. |

---

## 3. Reference: Mathematical Rationale

### Why 640 Dimensions?
*   **Divisibility**: We need $d_{model}$ to be divisible by $n_{head}$ (10) and hardware efficient (32).
*   **Capacity**: Empirical scaling laws (Kaplan et al., Hoffmann et al.) suggest ~640-768 for ~100M param models.

### Why 1728 MLP Hidden Size?
*   **The Problem**: SwiGLU uses 3 matrices ($W_{gate}, W_{val}, W_{down}$) vs standard FFN's 2 matrices ($W_{up}, W_{down}$).
*   **The Constraint**: Maintain same parameter count ($Params \approx 2 \times d \times 4d$).
*   **The Solution**: $3 \times d \times d_{swiglu} \approx 2 \times d \times 4d \rightarrow 3 d_{swiglu} \approx 8d \rightarrow d_{swiglu} \approx \frac{8}{3}d$.
*   **Calculation**: $640 \times \frac{8}{3} = 1706.66 \rightarrow$ Rounded to nearest 32 $\rightarrow$ **1728**.

### Why GQA (10 Heads / 5 KV Heads)?
*   **MHA (10/10)**: High quality, but stores 2 sets of full weights per token. Bandwidth heavy.
*   **MQA (10/1)**: Low memory, but reduces expressivity.
*   **GQA (10/5)**: The "Goldilocks" zone. Retains ~95% of MHA performance while cutting KV cache memory bandwidth by 50% (Ainslie et al., 2023).

### Why Shared Embeddings (Weight Tying)?
*   **The Concept**: The Input Embedding matrix ($W_{in}$) and the Output Classification Head ($W_{out}$) use the **exact same parameters**.
*   **Implementation**: `self.output.weight = self.embeddings.weight`.
*   **The Math**:
    *   Input: $Emb = \text{Lookup}(W, \text{indices})$
    *   Output: $Logits = x \cdot W^T$ (The matrix is transposed to project back to vocabulary).
*   **Rationale for Wolfgang-LM**:
    *   **Parameter Efficiency**: For a model this size (640 dim, 32k vocab), the embeddings are $\approx$ 21M parameters.
    *   **Impact**:
        *   **Untied**: Total Parameters $\approx$ 97M (Embeddings are $\approx$ 43% of model).
        *   **Tied**: Total Parameters $\approx$ 75M (Embeddings are $\approx$ 28% of model).
    *   **Intuition**: "Reading" (Input) and "Speaking" (Output) should use the same dictionary. Tying regularizes the model by enforcing that the vector representation of a token is the same whether it is being consumed or produced.

# GRU Update Operator — Learned Matching Prior

**File:** `devo/enet.py` — `class Update(nn.Module)`

---

## Overview

The `Update` operator is the core learning component of eVONet. It is called **18 times per
forward pass** (STEPS=18 during training), each time refining the hidden state `net` and
producing a flow correction `delta` and confidence `weight` that drive differentiable BA.

It is more than a flow predictor — it performs **learned approximate inference** over patch
correspondences, combining:

1. **Wide-basin correlation search** — sees a ±12px window, not just one point
2. **Temporal memory** — accumulates evidence across 18 iterations via GRU gating
3. **Spatial consensus** — propagates confident matches to ambiguous neighbours
4. **Learned outlier rejection** — predicts low confidence for unreliable patches

These four properties together constitute the **learned matching prior** — what is lost if
the indirect method is replaced with a direct photometric method.

---

## Hidden State `net`

```
Shape: (B, num_edges, 384)
```

The hidden state persists across all STEPS calls. It is the **belief state** of a learned
Kalman filter:

- **Iteration 1** (poor pose estimate): correlation is noisy, `net` ≈ 0, `delta` is small
  and uncertain, `weight` is low.
- **Iteration 10+** (refined pose): correlation has a sharp peak, `net` has accumulated 10
  rounds of evidence, `delta` is confident, `weight` is high.

Over training (BPTT across 18 iterations), different dimensions of the 384-dim state
specialise to encode: flow direction/magnitude, match confidence, local scene structure
(depth, planarity), camera motion pattern, etc. This specialisation is not programmed — it
emerges from the gradient signal of the pose and flow losses.

---

## Forward Pass — Step by Step

### Step 1: Fuse Three Information Sources

```python
net = net + inp + self.corr(corr)
net = self.norm(net)
```

Three signals are additively fused into `net`:

| Signal | Source | Role |
|--------|--------|------|
| `net[t-1]` | Previous hidden state | Memory of all prior iterations |
| `inp` | `imap[:, kk]` from inet | Static patch context (never changes per iteration) |
| `corr(corr)` | CorrBlock output through MLP | Current correlation evidence at this pose estimate |

**`corr` MLP** receives `(2 × 49 × p²)` similarity scores — 2 pyramid scales × 7×7 search
window × 3×3 patch pixels. It learns to read correlation peaks:

> *"Sharp peak at search offset (+3, +2) → encode strong (+3, +2) flow signal into net."*

The **±12px effective search radius** (scale=4 × radius=3) is the key advantage over direct
methods. Even if the current pose estimate is 10px off, the correct match is still inside the
window and the MLP can learn to find it. A direct method evaluating at one point has no such
convergence basin.

**`LayerNorm`** after fusion prevents any one source from dominating and stabilises gradients
during backpropagation through 18 iterations (BPTT).

---

### Step 2: Neighbor Message Passing

```python
ix, jx = fastba.neighbors(kk, jj)
mask_ix = (ix >= 0).float().reshape(1, -1, 1)
mask_jx = (jx >= 0).float().reshape(1, -1, 1)

net = net + self.c1(mask_ix * net[:, ix])   # cross-frame: same patch
net = net + self.c2(mask_jx * net[:, jx])   # cross-patch: same frame pair
```

`fastba.neighbors(kk, jj)` returns for each edge `e = (patch_k → frame_j)`:
- `ix[e]`: index of edge `(patch_k → frame_j')` — **same patch, different target frame**
- `jx[e]`: index of edge `(patch_k' → frame_j)` — **different patch, same target frame**

Masks zero out boundary edges that have no valid neighbour.

#### `c1` — Cross-Frame Consistency (same patch)

```
Linear(384) → ReLU → Linear(384)
```

**What it learns:** If patch `k` is confidently tracked to frame `j'` (strong correlation,
high-magnitude `net` state), propagate that into the state for `(patch k → frame j)`.

Geometrically: if you know where patch `k` is in frame `j'` reliably, and the relative pose
between frames `j'` and `j` is becoming consistent, you should be more confident about where
it is in frame `j`.

Training via BPTT teaches `c1` which cross-frame neighbour states actually improve flow
accuracy and which to ignore.

#### `c2` — Spatial Consensus (same frame pair)

```
Linear(384) → ReLU → Linear(384)
```

**What it learns:** If patches 42, 43, 44 all tracked to frame `j` agree on motion (+3, 0),
pull the state of ambiguous patch 45 toward that consensus.

This is a **learned spatial consistency check** — analogous to RANSAC but fully
differentiable. A single outlier patch cannot dominate because the other patches' consensus
propagates through `c2` and corrects its hidden state.

---

### Step 3: Soft Attention Pooling (`SoftAgg`)

```python
# SoftAgg internals:
w = scatter_softmax(self.g(x), group_index, dim=1)   # attention weights (learned)
y = scatter_sum(self.f(x) * w, group_index, dim=1)   # weighted sum per group
return self.h(y)[:, group_index]                      # broadcast back to each edge
```

Three linear layers per `SoftAgg`:
- **`g`**: scores each edge's hidden state for importance (attention query)
- **`f`**: transforms hidden state before pooling (value projection)
- **`h`**: transforms pooled result before adding back (output projection)

#### `agg_kk` — Pool over all edges for the same patch

Groups: all edges `(patch_k → frame_0)`, `(patch_k → frame_1)`, ..., `(patch_k → frame_n)`

**What it learns:** Which target frames provide reliable evidence about patch `k`'s depth and
position? Target frames with a strong, consistent correlation peak get high attention weight
and dominate the pooled signal. Ambiguous frames are suppressed without hard thresholding.

The pooled result `y` represents the **global tracking consensus for patch `k`**, broadcast
back to every edge of that patch.

#### `agg_ij` — Pool over all edges for the same frame pair

Groups: all patches tracked between the same pair of frames `(i, j)`

**What it learns:** Which patches are most informative about the relative pose between frames
`i` and `j`? High-gradient, distinctively-matched patches dominate. This gives a **global
motion estimate for the frame pair**, shared across all its patches — suppressing outlier
patches without hard rejection.

---

### Step 4: GRU-Style Gated Update

```python
self.gru = nn.Sequential(
    nn.LayerNorm(dim, eps=1e-3),
    GatedResidual(dim),   # gate = sigmoid(W*net); out = net + gate * res(net)
    nn.LayerNorm(dim, eps=1e-3),
    GatedResidual(dim),
)
```

Each `GatedResidual` computes:

```python
gate = sigmoid(Linear(net))          # ∈ (0,1) per dimension
res  = Linear(ReLU(Linear(net)))     # proposed state update
out  = net + gate * res              # gated residual
```

**Per-dimension selectivity:**
- Dimensions encoding **confident match direction**: `gate ≈ 1` → large update → state
  rapidly incorporates the new correlation peak
- Dimensions encoding **uncertain / ambiguous regions**: `gate ≈ 0` → small update → prior
  belief is preserved, not overwritten by noisy correlation

This is the **temporal memory mechanism**. Unlike a simple running average, the gate learns
*which aspects* of the belief to update vs hold. After 18 iterations, the state has
selectively accumulated evidence from all prior correlation observations.

---

### Step 5: Predict Outputs

```python
weights = self.w(net)    # (B, num_edges, 2) ∈ (0,1) — confidence
delta   = self.d(net)    # (B, num_edges, 2) ∈ R      — flow correction
```

#### `delta` — Flow Correction δ_inj

Added to the current reprojected patch coordinates to form the **target** for BA:

```python
target = projected_coords[..., p//2, p//2, :] + delta
```

BA then minimises `||reprojection(Gs, patches) - target||²_weight`.

`GradientClip` limits gradient magnitude during BPTT to ±0.01, preventing gradient
explosion through 18 unrolled steps.

#### `weight` — Confidence Σ_inj (Learned Outlier Rejection)

Used as the weight matrix `W` in the weighted BA cost. The network learns:

| Scenario | What the network sees | What it predicts |
|----------|----------------------|------------------|
| Sharp correlation peak, high-gradient patch | Strong, consistent corr signal in `net` | High weight → patch strongly constrains BA |
| Flat correlation (featureless region) | Low-magnitude, diffuse corr in `net` | Low weight → patch barely affects BA |
| Outlier (wrong match, large residual) | Inconsistent with `c2` consensus | Low weight → patch suppressed |

This is fundamentally different from geometric outlier rejection (e.g., residual
thresholding): the network learns from **data** what event correlation responses look like
at true vs false matches, and encodes that as a continuous confidence score.

---

## How Training Teaches All of This (BPTT)

The loss at each iteration `t` flows backward through the entire computation graph:

```
L_flow[t]
  → delta[t]  (via self.d)
    → net[t]  (via GRU, agg_ij, agg_kk, c1, c2, norm)
      → corr(corr[t])         ← teaches corr MLP to read correlation peaks
      → c1(net[t-1, ix])      ← teaches c1 which cross-frame messages help
      → c2(net[t-1, jx])      ← teaches c2 which cross-patch messages help
      → agg_kk, agg_ij        ← teaches attention which edges to up-weight
      → GatedResidual gates    ← teaches which dimensions to update vs hold
      → net[t-1]              ← recurse back through all previous iterations
        → ... → net[0]
```

Because `L_flow` is evaluated **at every iteration** (not just the last), the network is
penalised for slow convergence. This trains it to produce a good estimate after just 3–4
iterations, not only after 18.

---

## Why This Cannot Be Replaced by a Direct Method

A direct photometric method evaluates the feature difference at **one point** (the current
projection). Replacing the Update operator with gradient descent on photometric error loses:

| Property | Update Operator | Direct Method |
|----------|----------------|---------------|
| Search basin | ±12px (correlation window) | ~2px (gradient valid near minimum) |
| Temporal memory | 18-iteration hidden state | None — each step memoryless |
| Spatial consensus | c1/c2 message passing + SoftAgg | Only through shared BA variables |
| Outlier rejection | Learned weight prediction | Geometric only (residual magnitude) |
| Appearance prior | Trained on event correlation patterns | Raw feature difference, no prior |

The convergence basin issue is the most critical for event data: without good pose
initialisation (which is harder for events than RGB due to appearance variability), the
photometric gradient points in the wrong direction and the optimisation diverges.

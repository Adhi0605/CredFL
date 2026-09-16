# Adaptive Differential Privacy for Federated Fraud Detection — Project Roadmap

## Project Statement

Federated Learning (FL) allows multiple institutions to train models collaboratively without sharing raw data. Adding Differential Privacy (DP) noise protects that data further, but fixed-noise DP either over-protects (hurting accuracy) or under-protects (risking leakage) — especially under non-IID data, as is typical across financial institutions detecting fraud. This project builds an **adaptive DP mechanism** within FL that adjusts noise dynamically to preserve both privacy guarantees and model performance.

---

## Phase 1: Foundation


### Goals
- Understand the DP-FL landscape and lock in tools/data before writing code.

### Implementation Steps
1. **Pick a framework** — Flower (`flwr`) is the easiest for custom FL logic + DP hooks; alternatively TensorFlow Federated (has native DP support) or a hand-rolled PyTorch loop for full control over the adaptive mechanism.
2. **Pick a dataset** — European Credit Card Fraud dataset (Kaggle, ~284k transactions, heavily imbalanced) or IEEE-CIS Fraud Detection dataset. Either works; the European CC dataset is smaller and easier to iterate on.
3. **Define "institutions"** — since you don't have real multi-bank data, simulate clients by partitioning the single dataset (see Phase 2).

### Deliverable
- Environment set up (Python, Flower/PyTorch, Opacus for DP)
- Dataset downloaded and understood (class imbalance ratio, feature schema)
- Short written summary of chosen adaptive strategy direction (client/round/gradient-adaptive)

---

## Phase 2: Baseline System


### Goals
- Establish the two reference points your final method must sit between: no-DP accuracy (ceiling) and fixed-DP accuracy (floor).

### Implementation Steps
1. **Non-IID partitioning** — split the dataset across N simulated clients using a Dirichlet distribution (`α` controls heterogeneity severity — small α = more skewed). Vary fraud ratio and volume per client to mimic real institutions.
2. **Vanilla FedAvg** — implement standard federated averaging with no DP. Train, evaluate, record accuracy/F1/AUPRC.
3. **Fixed-noise DP-FedAvg** — add DP-SGD (via Opacus or manual gradient clipping + Gaussian noise) with a static noise multiplier σ and static ε target. Train at 2–3 different fixed ε values (e.g., ε = 1, 5, 10) to show the tradeoff curve.
4. **Log everything** — accuracy, precision/recall/F1, AUPRC, and cumulative ε per configuration. This becomes your baseline comparison table.

### Deliverable
- Working FedAvg pipeline (with and without static DP)
- Baseline results table/plot: accuracy vs. fixed ε, across at least 2 non-IID severity levels

---

## Phase 3: Core Contribution — Adaptive Noise Mechanism


### Goals
- Build and justify the adaptive DP mechanism. This is the novel part of the project — be precise about the *rule* and *why* it should outperform static noise.

### Implementation Steps
1. **Choose your adaptive axis** (pick one as primary, others as extensions):
   - **Client-adaptive**: scale noise per client based on local dataset size and/or data quality — small/noisy clients get proportionally different clipping/noise so they don't dominate or degrade the global model.
   - **Round-adaptive**: start with higher noise (more privacy pressure early) and decay it as training converges, since early gradients are less informative anyway.
   - **Gradient-adaptive**: track running statistics of gradient norms per round and adjust the clipping threshold + noise scale dynamically instead of using a fixed clip value.
2. **Implement the adaptation rule** as a modular function that plugs into the DP-SGD step (so it can be swapped/ablated later).
3. **Privacy accounting** — implement RDP or PRV accounting that correctly tracks *cumulative* privacy loss across rounds, especially important since noise now varies per round/client. Validate this against Opacus's built-in accountant on a simple case before trusting your own numbers.
4. **Sanity-check on a toy example first** (e.g., MNIST or a small synthetic set) before running on the fraud dataset — adaptive DP bugs are easy to hide in a highly imbalanced, noisy dataset.

### Deliverable
- Adaptive DP-FL implementation, integrated into the Phase 2 pipeline
- Verified privacy accounting (cumulative ε matches expected bounds)
- Initial results on the fraud dataset showing adaptive vs. fixed-noise performance at matched privacy budgets

---

## Phase 4: Evaluation


### Goals
- Rigorously demonstrate the privacy-utility tradeoff improvement.

### Implementation Steps
1. **Metrics** — don't rely on accuracy alone (fraud is rare/imbalanced): report Precision, Recall, F1, and AUPRC. Also report confusion matrix at a fixed operating threshold.
2. **Tradeoff curves** — plot accuracy/F1 vs. ε for: (a) no DP, (b) fixed-noise DP, (c) your adaptive method. Do this across multiple non-IID severity levels (mild, moderate, extreme Dirichlet α).
3. **Ablations** — if your adaptive rule has multiple components (e.g., both round-decay and per-client scaling), test each in isolation to show which part actually drives the improvement.
4. **Robustness checks** — vary number of clients, client participation rate per round, and local epoch count to confirm the method isn't overfit to one configuration.

### Deliverable
- Full results section: tables + tradeoff plots comparing all three methods across heterogeneity levels
- Ablation table isolating the contribution of each adaptive component

---

## Phase 5: Writeup & Packaging
**Duration:** ~1–2 weeks

### Goals
- Turn results into a presentable, defensible deliverable (report/paper and/or portfolio piece).

### Implementation Steps
1. **Report structure** — Problem statement → Related work → Method (your adaptive rule, formally stated) → Experimental setup → Results → Limitations → Conclusion.
2. **Limitations section** — be upfront about: simulated vs. real multi-institution data, accounting method assumptions, computational overhead of adaptive noise calculation.
3. **Repo cleanup** (if this is a portfolio piece) — clear README with problem statement, architecture diagram, how to reproduce results, and a summary table of key findings.
4. **Optional: architecture diagram** — visualize the FL round loop showing where the adaptive noise decision happens (useful both for the report and for interviews).

### Deliverable
- Final report/paper
- Clean, documented GitHub repository
- (Optional) architecture diagram and results summary for portfolio/interview use

---

## Key Risks to Track Throughout

| Risk | Mitigation |
|---|---|
| Non-IID fraud data may be *extremely* skewed (fraud is inherently rare and unevenly distributed) | Budget extra debugging time; this is expected and actually strengthens your motivating case for adaptive DP |
| Silent bugs in custom privacy accounting | Cross-validate against Opacus's accountant before trusting custom RDP/PRV implementations |
| Adaptive mechanism adds computational overhead per round | Track wall-clock time as a secondary metric; note this as a limitation/tradeoff in the report |
| Overfitting the adaptive rule to one dataset/config | Test across multiple heterogeneity levels and client counts before claiming generality |

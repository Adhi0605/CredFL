# Adaptive Differential Privacy for Federated Credit Card Fraud Detection
## Project Framework, Strategic Map & 7–8 Week Timeline

**Team:** Adhi Jeganathan Nageswaran, Edwin Sajith Ainikkal, Aaron Premji, Vrishank Umrani
**Dataset:** `creditcard.csv` — 284,807 transactions, 28 anonymized features (V1–V28) + Time + Amount, with a `Class` label (0 = normal, 1 = fraud). Fraud is extremely rare (~0.17% of all transactions).

---

## 1. The Big Idea, In Plain English

Imagine four banks. Each one has its own list of credit card transactions, and each list has some fraud hidden in it. No bank is allowed to send its raw transaction data to anyone else — that's private, sensitive information.

But all four banks *want* to build one smart fraud-detection model together, because a model that learns from everyone's data is better at catching fraud than four separate models trained alone.

**Federated Learning (FL)** is the trick that makes this possible: instead of sending data, each bank trains a small model on its own machine and only sends the *model's learned adjustments* (not the data itself) to a central server, which blends everyone's adjustments into one strong global model.

The problem: even those "adjustments" can leak secrets if someone studies them closely enough. So we add **Differential Privacy (DP)** — controlled mathematical noise — to hide individual transactions inside the adjustments.

The catch with DP: if you add the *same amount* of noise everywhere, you drown out the rare fraud signals (since fraud is already only 0.17% of the data). Add too little noise, and privacy breaks. Add too much, and the model goes blind to fraud.

**Our project's actual contribution:** an *adaptive* version of DP that adds **more noise where it's safe to do so, and less noise around the parts of the model that matter most for catching fraud** — so we keep both privacy and accuracy.

---

## 2. Strategic Map (How All the Pieces Connect)

```
 [Bank 1 data] [Bank 2 data] [Bank 3 data] [Bank 4 data]   <- simulated by splitting creditcard.csv
        |             |             |             |
   shared fitted preprocessing (training data only)
   Amount: log1p + standardize; Time: standardize
        |             |             |             |
   local training  local training  local training  local training
        |             |             |             |
   gradient clipping (EMA-based, adaptive)
        |
   noise injection (Laplace / Gaussian, adaptive via Fisher information)
        |
        └──────────────┬──────────────┘
                        |
              SECURE AGGREGATION (server)
                        |
              PRIVACY BUDGET TRACKER (privacy filter)
                        |
              GLOBAL FRAUD DETECTION MODEL
                        |
              EVALUATION (accuracy vs. privacy vs. baseline)
```

**Reading the map:** data never leaves each simulated "client." Only clipped, noised model updates travel to the server. The server tracks how much total privacy has been "spent" and adjusts future noise accordingly. The output is one shared model, tested for both fraud-catching power and privacy strength.

---

## 3. Clear Definitions of Every Part We're Building

| Part | Simple Definition | Why It's In Our Project |
|---|---|---|
| **Federated Learning (FL)** | Training one shared model across several separate data-holders without moving the raw data. | Lets simulated "banks" collaborate without exposing transactions. |
| **Client / Node** | One simulated participant (e.g., a bank) holding a private slice of the dataset. | We'll split `creditcard.csv` into multiple non-identical slices to mimic real banks. |
| **Local Training** | Each client trains the model briefly on its own data slice. | Produces local model updates instead of sharing data. |
| **FedAvg (Federated Averaging)** | The baseline method for combining every client's update into one global model (a weighted average). | Our starting point / baseline before adding privacy and adaptivity. |
| **Differential Privacy (DP)** | A mathematical guarantee that adding or removing any single transaction barely changes the model's output — enforced by injecting random noise. | Protects individual transactions from being reverse-engineered out of the shared updates. |
| **Gradient** | The "direction of learning" a model computes after seeing data — essentially, how much each internal parameter should change. | This is exactly what gets noised and sent to the server. |
| **Gradient Clipping** | Capping how large any single gradient update is allowed to be, before noise is added. | Prevents one unusual transaction from dominating the update and controls how much noise is needed. |
| **EMA (Exponential Moving Average) Clipping** | Instead of a fixed clipping cap, the cap adjusts over time based on recent gradient history. | This is one of our *adaptive* mechanisms — it keeps clipping sensible as training progresses. |
| **Laplace / Gaussian Noise** | Two standard statistical "noise recipes" used to satisfy DP guarantees. | The literal privacy mechanism — we choose/blend between them based on sensitivity. |
| **Adaptive Noise Injection** | Instead of one fixed noise level for the whole model, noise strength is varied per-parameter or per-round. | The core novelty of the project — protects privacy without blinding the model to rare fraud patterns. |
| **Fisher Information (layer-wise)** | A statistical measure of how *important* a specific model parameter is for making correct predictions. | Used to decide where noise can be applied heavily (low-importance parameters) vs. lightly (high-importance, fraud-detecting parameters). |
| **Secure Aggregation** | A protocol ensuring the server can compute the combined average of client updates without ever seeing any individual client's raw update. | Adds a second layer of protection on top of DP. |
| **Privacy Budget (ε, "epsilon")** | A running total of how much privacy has been "spent" through noise so far; once it's used up, no more training rounds are privacy-safe. | We track this across rounds so we can report a formal privacy guarantee, not just a description. |
| **Privacy Filter** | A mechanism that monitors the privacy budget in real time and can adjust or halt training to stay within limits. | Keeps our system accountable and dynamic instead of "set and forget." |
| **Class Imbalance** | The situation where one outcome (fraud) is vastly rarer than the other (normal transactions) — here, ~0.17% vs ~99.83%. | Directly affects both model accuracy and how noise should be distributed, since fraud signals are already faint. |
| **Training Preprocessing** | A single transformation fitted only on the combined client-training rows: apply `log1p` and then standardization to `Amount`, standardize raw elapsed `Time`, and leave `V1`–`V28` unchanged. | Prevents large, skewed values from dominating model gradients while avoiding leakage from the global test set. The fitted parameters are saved and reused unchanged in every experiment. |
| **SMOTE (Synthetic Minority Oversampling)** | A technique that creates synthetic, realistic-looking fraud examples to balance the training data locally. | Used at each client to stop the model from ignoring fraud entirely. |
| **Non-IID Data** | "Not Independently and Identically Distributed" — each client's data looks statistically different from the others (e.g., different fraud rates). | We'll simulate this deliberately, since it's realistic and stress-tests our adaptive method. |
| **Global Model** | The final, single fraud-detection model produced after all training rounds. | This is literally our deliverable — the thing we evaluate and report on. |
| **Evaluation Metrics** | Precision, Recall, F1-score, AUC-ROC — measures of how well the model catches fraud without too many false alarms. | Standard for imbalanced classification; accuracy alone is misleading here. |
| **Privacy-Utility Tradeoff** | The relationship between how much noise you add (privacy) and how much accuracy you lose (utility). | The central question our whole project tries to answer better than fixed-noise DP. |

---

## 4. Strategic Phases (The "Why" Behind the Weeks)

Think of the project in **five strategic phases**. Each week below maps into one of these:

1. **Foundation** — understand the data, understand the math, set up tools.
2. **Baseline** — build plain Federated Learning first (no privacy yet), so you have something to compare against.
3. **Privacy Layer** — add standard, fixed-noise Differential Privacy on top of the baseline.
4. **Adaptive Layer** — replace fixed noise with our adaptive mechanism (clipping + Fisher-weighted noise + privacy budget tracking).
5. **Proof & Reporting** — compare everything, prove the adaptive version is better, and write it up.

---

## 5. 7–8 Week Timeline

| Week | Phase | What We Do | Deliverable (Proof of Progress) |
|---|---|---|---|
| **Week 1** | Foundation | Explore `creditcard.csv` (check fraud rate, feature distributions, correlations). Set up Python environment (PyTorch/TensorFlow + Flower or PySyft for FL simulation). Split dataset into 4–6 simulated "client" partitions, some skewed to mimic non-IID banks. | Data exploration notebook + partitioned datasets ready to use. |
| **Week 2** | Foundation → Baseline | Fit the shared preprocessor on the union of the six client-training partitions only—never on `global_test.csv`. Apply `log1p` then `StandardScaler` to `Amount`; standardize raw elapsed `Time`; keep `V1`–`V28` and `Class` unchanged. Save the fitted metadata and parameters, transform datasets in memory, then build a simple centralized classifier (e.g., MLP or logistic regression). Confirm metrics (Precision/Recall/F1/AUC) look reasonable before adding complexity. | Saved preprocessing state + centralized baseline model + metrics table. |
| **Week 3** | Baseline | Implement plain Federated Learning with FedAvg across the simulated clients. No privacy yet. Compare its accuracy to the centralized baseline — expect it to be close but slightly lower. | Working FL simulation + FedAvg results. |
| **Week 4** | Privacy Layer | Add standard (fixed-noise) Differential Privacy to the FL pipeline — apply uniform gradient clipping and fixed Gaussian/Laplace noise. Measure how much accuracy drops. This *drop* is the problem we're solving. | Fixed-noise DP-FL results, showing the accuracy hit on fraud detection. |
| **Week 5** | Adaptive Layer (Part 1) | Implement EMA-based adaptive gradient clipping (clip limits adjust over training rounds instead of staying fixed). | Adaptive clipping module integrated and tested. |
| **Week 6** | Adaptive Layer (Part 2) | Implement Fisher-information-weighted noise injection — compute per-layer importance, apply less noise to fraud-critical parameters. Add the privacy budget tracker (privacy filter) to log cumulative epsilon spent. | Full Adaptive DP-FL system running end-to-end. |
| **Week 7** | Proof & Reporting | Run all three systems side-by-side under identical conditions: (1) plain FL, (2) fixed-noise DP-FL, (3) our Adaptive DP-FL. Compare Precision/Recall/F1/AUC and privacy budget spent. Build comparison charts/tables. | Final results table + charts proving adaptive DP beats fixed-noise DP at the same privacy level. |
| **Week 8** *(buffer / polish)* | Proof & Reporting | Write up methodology, results, and limitations. Polish visuals, prep the presentation/report, run any final ablation tests if time allows (e.g., testing 4 vs 8 clients). | Final report/paper draft + presentation-ready results. |

> **Note:** Week 8 is a buffer. If everything goes smoothly through Week 7, use Week 8 purely for polish, extra experiments, and a clean write-up. If Week 6 runs long, Week 8 absorbs the overflow — this is why the plan is framed as "7–8 weeks" rather than a rigid 7.

### Current implementation checklist

This checklist records work that is verifiably present in the repository. A checked item is implemented and, where applicable, has a generated artifact or a passing automated test. An unchecked item is still outstanding; partial weeks are split into individual tasks so completed groundwork is not overstated.

#### Week 1 - Foundation

- [x] Add the real `creditcard.csv` dataset under `data/raw/`.
- [x] Validate the expected schema, missing values, duplicate rows, class counts, fraud rate, feature correlations, and `Amount` distribution.
- [x] Create a leakage-safe, stratified 20% global test split (`data/processed/global_test.csv`).
- [x] Partition the remaining training data across six simulated clients using a Dirichlet non-IID split.
- [x] Enforce and verify a minimum of 10 fraud records per client.
- [x] Generate `dataset_summary.json`, `partition_manifest.csv`, and the six client CSV files.
- [x] Document client sizes, fraud prevalence, and non-IID observations in `CLIENT_DATA_STATS.md`.
- [x] Provide a repeatable Week 1 command-line workflow and an exploration notebook.
- [x] Configure the Python project with PyTorch, Flower, Opacus, scikit-learn, pandas, plotting, Jupyter, and test dependencies.
- [ ] Execute the exploration notebook end-to-end and save its cell outputs and generated charts (the notebook is present but currently unexecuted).

#### Week 2 - Preprocessing and centralized baseline

- [x] Implement a reusable `FraudPreprocessor` fitted on training data only.
- [x] Implement the default `Amount` transformation: `log1p` followed by standardization.
- [x] Implement standardization of raw elapsed `Time` while leaving `V1`-`V28` unchanged.
- [x] Preserve the `Class` target unchanged during transformation.
- [x] Implement the optional cyclical-time ablation (`time_sin` and `time_cos`).
- [x] Implement serialization and loading of preprocessing metadata and fitted parameters.
- [x] Add automated tests for default preprocessing, cyclical time, target/feature preservation, and saved-state reproducibility.
- [x] Fit the shared preprocessor on the combined six client-training partitions and save `artifacts/preprocessor.json`.
- [x] Build and train the centralized PyTorch MLP classifier (`30 -> 64 -> 32 -> 1`).
- [x] Reserve a stratified 10% slice of client-training data for decision-threshold selection without using the global test set.
- [x] Save the trained model state, run configuration, epoch history, and selected threshold under `artifacts/`.
- [x] Report centralized Precision, Recall, F1, AUC-ROC, and average-precision metrics in `reports/week2_centralized_metrics.md`.

#### Week 3 - Plain federated-learning baseline

- [x] Implement local client training across the six simulated clients.
- [x] Implement weighted FedAvg aggregation with no privacy mechanism.
- [x] Run the plain FL experiment under a recorded, reproducible configuration.
- [x] Compare plain FL metrics with the centralized baseline.

#### Week 4 - Fixed-noise DP-FL

- [ ] Add uniform gradient/update clipping.
- [ ] Add fixed Gaussian and/or Laplace noise to the federated pipeline.
- [ ] Track the privacy parameters used by the fixed-noise experiment.
- [ ] Run fixed-noise DP-FL and quantify its fraud-detection utility loss.

#### Week 5 - Adaptive clipping

- [ ] Implement EMA-based adaptive clipping.
- [ ] Integrate adaptive clipping into local training.
- [ ] Test and log how the clipping threshold changes during training.

#### Week 6 - Adaptive noise and privacy accounting

- [ ] Compute layer-wise Fisher information or the selected parameter-importance estimate.
- [ ] Implement importance-weighted adaptive noise injection.
- [ ] Implement cumulative epsilon accounting and a privacy filter.
- [ ] Run the complete Adaptive DP-FL pipeline end-to-end.

#### Week 7 - Comparative evaluation

- [ ] Run plain FL, fixed-noise DP-FL, and Adaptive DP-FL under identical data splits, preprocessing, seeds, and training settings.
- [ ] Compare Precision, Recall, F1, AUC-ROC, and privacy budget across all systems.
- [ ] Produce the final comparison table and charts.
- [ ] Verify whether Adaptive DP-FL outperforms fixed-noise DP-FL at a matched privacy level.

#### Week 8 - Reporting and polish

- [ ] Write the methodology, experiment configuration, results, limitations, and conclusions.
- [ ] Prepare presentation-ready figures and slides.
- [ ] Run final ablations, such as changing the number of clients, if time permits.
- [ ] Complete the final report or paper draft.

**Verification snapshot:** the repository currently passes all twelve automated data-partitioning, preprocessing, centralized-MLP, and federated-learning tests. The completed Week 2 MLP achieved Precision 0.803922, Recall 0.836735, F1 0.820000, AUC-ROC 0.975160, and Average Precision 0.698648 on the untouched global test set. The completed Week 3 plain FedAvg model achieved Precision 0.813187, Recall 0.755102, F1 0.783069, AUC-ROC 0.908559, and Average Precision 0.624199 on the same test set. Its accuracy was 0.999280 versus 0.999368 for the centralized baseline. Weeks 4-8 have no implementation or result artifacts yet, so their tasks remain unchecked.

### Preprocessing execution rule

`src/preprocessing.py` is a reusable training component, not a standalone data-splitting command. Use it after `global_test.csv` and the six client partitions have been created, and immediately before the first model-training experiment:

1. Load and combine the six client-training partitions only to fit one shared `FraudPreprocessor`.
2. Save the fitted transformation choice, scaler parameters, and final feature ordering (for example, in `artifacts/preprocessor.json`).
3. Transform each client partition in memory for local training; do not overwrite the existing client CSV files.
4. Transform `global_test.csv` with the same saved preprocessor for evaluation. Never fit or refit on the global test set.
5. Reuse exactly the same saved preprocessor for the centralized, plain FL, fixed-DP FL, and adaptive-DP FL experiments so their results remain directly comparable.

The default treatment is `Amount → log1p → standardization` and raw elapsed `Time → standardization`. An optional time ablation may replace raw `Time` with sine/cosine features calculated from `Time % 86400`; it must be recorded as a separate experimental configuration.

---

## 6. Suggested Role Split (Optional, for a 4-Person Team)

| Role | Focus | Roughly Maps To |
|---|---|---|
| Data & Baseline Lead | Data exploration, partitioning, centralized + FedAvg baseline | Weeks 1–3 |
| Privacy Mechanism Lead | Fixed-noise DP implementation, then adaptive clipping | Weeks 4–5 |
| Adaptive/Fisher Lead | Fisher-information noise weighting, privacy budget tracker | Weeks 6 |
| Evaluation & Writing Lead | Metrics, comparison charts, report/paper assembly | Weeks 7–8 |

All four should stay involved in every phase — this split just assigns *ownership*, not isolation, so nobody is blocked waiting on someone else.

---

## 7. Quick Risk Checklist (Things That Commonly Go Wrong)

- **Fraud is too rare to see in small client partitions** → make sure every simulated client keeps at least a handful of fraud cases, or the model can't learn from that client at all.
- **Privacy budget runs out before training finishes** → track epsilon from Week 4 onward, don't leave it until Week 7.
- **"Adaptive" looks the same as "fixed noise" in results** → make sure the Fisher-information weighting is actually changing noise levels differently across layers; log and plot it to confirm.
- **Federated simulation is slow** → keep local epochs per round low (1–3) especially while debugging; increase only once the pipeline works end-to-end.

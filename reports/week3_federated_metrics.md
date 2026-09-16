# Week 3 Plain Federated-Learning Results

Six clients participated in every one of 10 synchronous rounds,
training for 1 local epoch(s) per round. The server applied
sample-count-weighted FedAvg. No clipping, noise, differential privacy, or
secure aggregation was used. The decision threshold came only from held-out
client-training rows; `global_test.csv` remained evaluation-only.

| Metric | Centralized | Plain FedAvg | Difference (FL - centralized) |
|---|---:|---:|---:|
| Accuracy | 0.999368 | 0.999280 | -0.000088 |
| Precision | 0.803922 | 0.813187 | +0.009265 |
| Recall | 0.836735 | 0.755102 | -0.081633 |
| F1 | 0.820000 | 0.783069 | -0.036931 |
| AUC-ROC | 0.975160 | 0.908559 | -0.066601 |
| Average precision | 0.698648 | 0.624199 | -0.074450 |

FedAvg decision threshold: 0.924549.
Predicted fraud cases: 91
of 56,962 test rows.

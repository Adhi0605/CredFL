# Week 2 Centralized MLP Results

The model-development data came from the union of all six client-training
partitions, with 10% reserved for threshold selection. The shared preprocessor
was fitted only on client-training rows; `global_test.csv` was used only once
for final evaluation.

| Metric | Value |
|---|---:|
| Test rows | 56,962 |
| Test fraud cases | 98 |
| Decision threshold | 1.00 |
| Accuracy | 0.999368 |
| Precision | 0.803922 |
| Recall | 0.836735 |
| F1 | 0.820000 |
| AUC-ROC | 0.975160 |
| Average precision | 0.698648 |
| Predicted fraud cases | 102 |

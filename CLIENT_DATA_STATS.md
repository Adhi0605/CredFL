# Client Local Training Data Statistics

This report summarizes the six non-IID client datasets used as simulated local bank data. These partitions contain only training records; the global test set was separated before partitioning.

## Partition configuration

| Setting | Value |
|---|---:|
| Number of clients | 6 |
| Dirichlet alpha | 0.5 |
| Global test split | 20% |
| Random seed | 42 |
| Minimum fraud records per client | 10 |

## Client statistics

| Client | Local training rows | Share of training data | Normal transactions | Fraud transactions | Fraud percentage | Normal percentage |
|---|---:|---:|---:|---:|---:|---:|
| Client 0 | 5,093 | 2.24% | 4,855 | 238 | 4.6731% | 95.3269% |
| Client 1 | 123,513 | 54.20% | 123,495 | 18 | 0.0146% | 99.9854% |
| Client 2 | 35,779 | 15.70% | 35,747 | 32 | 0.0894% | 99.9106% |
| Client 3 | 2,839 | 1.25% | 2,806 | 33 | 1.1624% | 98.8376% |
| Client 4 | 34,910 | 15.32% | 34,865 | 45 | 0.1289% | 99.8711% |
| Client 5 | 25,711 | 11.28% | 25,683 | 28 | 0.1089% | 99.8911% |
| **Total** | **227,845** | **100.00%** | **227,451** | **394** | **0.1729%** | **99.8271%** |

## Observations

- The local datasets are strongly non-IID in both client size and fraud prevalence.
- Client 1 holds the majority of the training data (54.20%) but has the lowest fraud percentage (0.0146%).
- Client 0 has the most fraud records (238) and the highest fraud percentage (4.6731%).
- Client 3 is the smallest client, with 2,839 records, but has the second-highest fraud percentage (1.1624%).
- Every client satisfies the configured minimum of 10 fraud records.
- Across all clients, there are 394 fraud records among 227,845 local training records.

## Source

Statistics are derived from `data/processed/partition_manifest.csv`, generated using the real European credit-card fraud dataset.

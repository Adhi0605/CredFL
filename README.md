# CredFL

Adaptive differential privacy for federated credit-card fraud detection.

## Week 1: foundation

The first-week workflow validates and explores the European credit-card fraud
dataset, creates a leakage-safe global test set, and partitions only the training
rows among six simulated banks with a Dirichlet non-IID split.

### 1. Add the dataset

Download the Kaggle `creditcard.csv` dataset and place it at:

```text
data/raw/creditcard.csv
```

The file is intentionally ignored by Git. Expected columns are `Time`, `V1` to
`V28`, `Amount`, and binary target `Class`.

### 2. Create the environment

On Windows PowerShell with `uv` installed:

```powershell
$env:UV_CACHE_DIR="$PWD\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR="$PWD\.uv-python"
uv python install 3.12
uv sync --extra dev
```

### 3. Run Week 1

For an executable report, open `notebooks/01_week1_foundation.ipynb`. For a
repeatable command-line run:

```powershell
uv run python -m src.prepare_data --input data/raw/creditcard.csv
```

Outputs are written to `data/processed/`:

- `global_test.csv`: untouched, stratified test set for all future comparisons
- `client_00.csv` ... `client_05.csv`: non-IID client training partitions
- `partition_manifest.csv`: client sizes and fraud rates
- `dataset_summary.json`: schema, quality, imbalance, and correlation summary

Defaults: 20% global test data, 6 clients, Dirichlet alpha 0.5, seed 42, and at
least 10 fraud samples per client. Smaller alpha means stronger client skew.

## Training preprocessing

`FraudPreprocessor` provides one fitted transformation that can be shared by the
centralized baseline, every federated client, and global-test evaluation. Fit it
on training rows only; never include `global_test.csv` when estimating its
parameters.

```python
from src.preprocessing import FraudPreprocessor

preprocessor = FraudPreprocessor(time_strategy="standard")
preprocessor.fit(training_data)
training_data = preprocessor.transform(training_data)
test_data = preprocessor.transform(test_data)
preprocessor.save("artifacts/preprocessor.json")
```

The default pipeline applies `log1p` followed by standardization to `Amount`,
standardizes raw elapsed `Time`, and leaves `V1` through `V28` unchanged. The
optional `time_strategy="cyclical"` replaces `Time` with sine and cosine features
computed from `Time % 86400`. The `Class` target is always preserved unchanged.

## Week 1 completion checklist

- [ ] Place the real `creditcard.csv` in `data/raw/`
- [ ] Run the notebook end-to-end and save its outputs
- [ ] Confirm all validation checks pass
- [ ] Review class imbalance, distributions, correlations, and client skew
- [ ] Commit code/notebook/manifest; do not commit raw or processed transaction CSVs

## Week 2: centralized MLP baseline

Train the global MLP on the union of the six client-training partitions and
evaluate it on the untouched global test set:

```powershell
uv run python -m src.centralized_baseline
```

The command fits the shared preprocessor on training rows only and writes:

- `artifacts/preprocessor.json`: reusable fitted preprocessing state
- `artifacts/centralized_mlp.pt`: trained PyTorch model parameters
- `artifacts/centralized_mlp_run.json`: architecture, training history, and metrics
- `reports/week2_centralized_metrics.md`: readable final metrics table

Defaults: hidden layers of 64 and 32 units with ReLU and 0.2 dropout, AdamW,
class-weighted binary cross-entropy, 10 epochs, batch size 2,048, and seed 42.
Ten percent of the client-training rows are held out to choose the F1-maximizing
decision threshold; the global test set is never used for threshold selection.

## Week 3: plain federated-learning baseline

Run synchronous FedAvg across all six simulated clients and compare the global
model with the Week 2 centralized baseline:

```powershell
uv run python -m src.federated_baseline
```

Every client starts each round from the same global MLP, trains locally, and
returns model parameters to the simulated server. The server averages parameters
in proportion to client training-set size. Defaults are 10 rounds, one local
epoch per round, full client participation, and the same optimizer, architecture,
preprocessor, batch size, and seed used for Week 2. This baseline deliberately
has no clipping, noise, differential privacy, or secure aggregation.

The run writes `artifacts/federated_mlp.pt`,
`artifacts/federated_mlp_run.json`, and
`reports/week3_federated_metrics.md`. Override common settings with `--rounds`,
`--local-epochs`, `--batch-size`, `--learning-rate`, `--seed`, and `--device`.

## Run on another laptop

### Prerequisites

- Install Git and uv (installation instructions: https://docs.astral.sh/uv/getting-started/installation/).
- Use Python 3.12; the project supports Python 3.11 and 3.12. uv can install Python for you.
- Internet access is needed for the initial dependency and dataset downloads.
- A CPU is sufficient. An NVIDIA GPU is optional; the commands below explicitly use CPU for portability. All six banks are simulated in one Python process on one laptop.

### Clone and install (Windows PowerShell, macOS, or Linux)

Clone this private repository using a GitHub account with access:

```sh
git clone https://github.com/Adhi0605/CredFL.git CredFL
cd CredFL
uv python install 3.12
uv sync --locked --extra dev --python 3.12
```

Run all commands from this directory, which contains `pyproject.toml`. `uv sync`
creates a local `.venv` and installs the versions in `uv.lock`; do not copy a
virtual environment from another laptop. The `--extra dev` option installs pytest
and Ruff as well as the runtime dependencies. No manual environment activation
is required when using `uv run`.

### Download the dataset

Download the European credit-card fraud dataset (`creditcard.csv`) from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
A Kaggle account or acceptance of the dataset's terms may be required. Extract
the CSV to `data/raw/creditcard.csv`, not inside another nested folder. The
expected schema is `Time`, `V1` through `V28`, `Amount`, and `Class`.

Raw transactions and generated client/test CSVs are excluded from Git. Each
laptop must download the source dataset and regenerate the partitions.

### Execute the complete implemented workflow

Run these commands in order:

```sh
uv run --locked python -m src.prepare_data --input data/raw/creditcard.csv
uv run --locked python -m src.centralized_baseline --device cpu
uv run --locked python -m src.federated_baseline --device cpu
```

1. Data preparation validates the CSV and creates six client partitions plus a
   held-out global test set in `data/processed/`.
2. Centralized training writes a fitted preprocessor, model, and run metadata to
   `artifacts/`, and metrics to `reports/week2_centralized_metrics.md`.
3. Federated training uses the centralized preprocessor and run metadata, then
   writes its model/run metadata and `reports/week3_federated_metrics.md`.

Run centralized training before federated training whenever you regenerate data
or change training settings. Existing committed artifacts and reports are saved
results from an earlier run; the commands above overwrite those outputs. Results
and runtime may vary by hardware. Differential privacy is planned work; the
current federated baseline does not provide differential privacy.

For a shorter functional run (not the default benchmark), use:

```sh
uv run --locked python -m src.centralized_baseline --device cpu --epochs 1
uv run --locked python -m src.federated_baseline --device cpu --rounds 1 --local-epochs 1
```

### Notebooks and checks

```sh
uv run --locked jupyter lab
uv run --locked pytest
uv run --locked ruff check .
```

Open `notebooks/01_week1_foundation.ipynb` in JupyterLab and use the project's
Python environment. The tests use synthetic data and do not require the Kaggle
CSV. For all CLI options, append `--help` to any Python module command above.

### Troubleshooting

- `uv` or `git` is not recognized: finish installing the tool and open a new terminal.
- Unsupported Python version: rerun `uv sync --locked --extra dev --python 3.12`.
- `No module named src`: run from the repository root and use `python -m src...`.
- Missing `creditcard.csv`: extract it at the exact path shown above.
- Missing client partitions: run `src.prepare_data` first.
- Missing preprocessor or centralized run JSON: run `src.centralized_baseline`
  before `src.federated_baseline`.
- CUDA unavailable: use `--device cpu`. `--device auto` selects CUDA when PyTorch
  detects it and otherwise uses CPU.
- Memory pressure during training: try `--batch-size 512` on both training commands.

### Repository contents

- `src/`: data preparation, preprocessing, centralized MLP, and FedAvg code.
- `tests/`: automated checks using synthetic inputs.
- `notebooks/`: exploratory Week 1 notebook.
- `artifacts/` and `reports/`: saved models, run metadata, and result summaries.
- `Project_Framework_and_Timeline.md`, `CLIENT_DATA_STATS.md`, and
  `federated-dp-fraud-detection-roadmap.md`: project planning and data notes.
- `pyproject.toml` and `uv.lock`: supported Python range and locked dependencies.

# Windowed PINN for the Lorenz-1960 System

This repository contains the code and archived numerical results used to study
full-interval and windowed physics-informed neural networks (PINNs) for the
Lorenz-1960 ordinary differential equation system.

The selected method trains one PINN per time window, applies causal residual
weighting within each window, transfers weights to the next window, and uses
Adam followed by L-BFGS refinement. The archive also contains the paper's
ablation arms and the full-interval baseline.

## Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

## Verify the archived results

The verification command checks the SHA-256 digest of every final checkpoint,
checks the full run-file manifest, validates the 14-row ablation table and optimizer histories, reloads all 16
models on CPU, and evaluates each model on the same 13,265-point DOP853
reference grid used for the archived summaries.

```bash
python scripts/verify_archive.py
```

The command does not train a model. It writes:

- `artifacts/checkpoint-evaluation.csv`, with archived and recomputed RMSE;
- `artifacts/verification.json`, with the evaluation environment and result;
- console output for each checkpoint.

Run the source regression tests with:

```bash
python -m pytest src/pinn/test_pinn.py src/pinn/test_stress.py
```

## Inspect one saved model

The candidate checkpoint can be evaluated without retraining:

```bash
python run_eval.py runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd
```

To print the archived ablation summary and its statistical comparisons:

```bash
python scripts/ablation_report.py
```

## Train the principal configurations

These commands are computationally expensive and are not part of verification.

```bash
python run_pinn.py
python run_causal.py
python scripts/paper_ablations.py --arms r1,r2,r4,r3 --seeds 1,2,3
```

`run_pinn.py` uses the central defaults. `run_causal.py` trains the selected
27-window causal configuration. The ablation driver is resume-safe when a
complete summary already exists. The recovered `sequential_run.py` is retained
for provenance but is incomplete and does not match the archived R6 settings;
it is not a supported retraining command. R6 checkpoint evaluation is verified.
A corrected reproduction runner requires a separate, explicitly recorded change.

## Archived experiment map

| Location | Paper role | Runs |
| --- | --- | ---: |
| `runs/4x60_f64_unit` | F1, full-interval PINN | 1 |
| `runs/causal-window/candidate-1536pt/...` | selected windowed causal run, seed 0 | 1 |
| `runs/paper_ablations/r1_no_causal_seed*` | causal weighting removed | 3 |
| `runs/paper_ablations/r2_no_warm_seed*` | warm start removed | 3 |
| `runs/paper_ablations/r3_causal_single_seed0` | causal training without windowing | 1 |
| `runs/paper_ablations/r4_candidate_seed*` | selected method, seeds 1-4 | 4 |
| `runs/paper_ablations/r6_sequential_seed*` | shared-network sequential baseline | 3 |

Every run includes its final checkpoint, numerical configuration, loss history,
evaluation summary, metrics, and available stage/mark records. Aggregate
material is under `runs/paper_ablations/`:

- `ablations.csv`: all 14 ablation rows;
- `physics_checks.csv` and `physics_checks.json`: invariant and closure checks;
- `PAPER_TABLES.md`: compact tables derived from the archived runs;
- `environment.jsonl`: recorded Colab execution environments.

## Repository layout

```text
src/baseline/       DOP853 reference solver and metrics
src/pinn/           PINN models, training, evaluation, and plots
scripts/            paper experiment, reporting, and verification commands
runs/               selected checkpoints and numerical results
artifacts/          integrity, provenance, and verification records
figures/            manuscript figures and preserved diagram source
```

## Provenance and limits

The archived Colab record identifies commit
`ed43703233264b5568ec4598dedc1cf84ac0f452` with `git_dirty=true`. The
corresponding uncommitted diff was not archived, so that commit alone cannot
certify the exact historical training source. The committed configurations are
the numerical authority for reconstructing each model, and the checkpoints are
verified directly against their saved summaries. The currently audited source
was recovered in commit `babc47aab9f7cdbe8fde832cdd1e670eab1f9210` and copied
from repository snapshot `36ccfea89c05400aeca5d774c5ec4ad871497d09`.

Some archived seed-0 runs were produced on a GPU and later runs include CPU and
GPU executions. Floating-point results may differ slightly across hardware and
library versions. The repository therefore verifies archived checkpoints within
an explicit numerical tolerance and does not promise byte-for-byte retraining
determinism.

Machine-specific output paths in archived `config.json` files were replaced by
repository-relative paths. `artifacts/config-provenance.json` records the
original file hash, published file hash, and exact path transformation for each
run. No license has been assigned to this repository.

## Figures

The three manuscript figures are in `figures/`. The pipeline PDF is unchanged
from the source document. `python scripts/pipeline_figure.py` regenerates its
standalone HTML/SVG scene, including the original print dimensions. Print the
HTML with browser headers disabled after MathJax has loaded.

The exact compact plotting scripts were not archived. Their data and canonical
plotting functions are available, and their original PDF sources are retained
in `figures/source/`. Font repair preserves the page content streams, axes,
curves, colors, labels, positions and page boxes. To reproduce it, install
`requirements-figures.txt`, copy the two PDFs from `figures/source/` to
`figures/`, then run `python scripts/embed_figure_fonts.py`. This substitutes
embedded TrueType outlines for the same DejaVu Type 3 glyphs. Figure geometry
is unchanged; small printed labels remain an acknowledged limitation.

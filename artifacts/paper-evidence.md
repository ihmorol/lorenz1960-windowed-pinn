# Paper evidence completeness

All 16 reported runs have final checkpoints, configurations, optimizer histories,
metrics and summaries. Completion means the recorded training stages finished;
a high final error is a completed failure case. It does not mean every claim
in the manuscript has been established.

| Paper experiment | Saved seeds | Runs | Artifact status |
| --- | --- | ---: | --- |
| Full-interval F1 | 0 | 1 | Complete; checkpoint error verified |
| Windowed candidate | 0-4 | 5 | Complete; checkpoint errors verified |
| No causal weighting | 1-3 | 3 | Complete; checkpoint errors verified |
| No warm start | 1-3 | 3 | Complete; checkpoint errors verified |
| Causal single network | 0 | 1 | Complete; checkpoint error verified |
| Shared-network sequential | 1-3 | 3 | Complete; checkpoint errors verified |

`verification.json` and `checkpoint-evaluation.csv` report the direct evaluation
of all 16 checkpoints against 13,265 DOP853 reference times. `archive.sha256`
covers all 150 archived run files. `ablation_report.py` reads the saved runs to
recompute table aggregates and statistical comparisons.

## Completed training source

The recorded training commit, `ed43703233264b5568ec4598dedc1cf84ac0f452`,
contains the completed shared-network implementation. Its ablation driver
matches every non-path setting of all 14 saved ablation configurations after
applying defaults for fields absent in older records. The completed code and
its eight existing regression tests are included. `source-check.json` records
an executable-AST comparison of the recovered files with that commit.

The obsolete standalone sequential prototype was removed. No new scientific
runner was invented and no full research experiment was rerun. The original
environment was marked dirty and its uncommitted diff was not saved, so the
exact historical working tree cannot be certified.

## Claims requiring manuscript clarification

- The stored accuracy evaluation uses DOP853. A separate RK4 comparison is not
  established by these verification records.
- Final shared-network errors support a poor final fit. Intermediate evaluation
  of earlier windows is needed to establish the claimed forgetting mechanism.
- Reproducing the reported p-value does not establish its statistical assumptions
  or a general causal-weighting advantage with only three to five seeds.
- The invariant pair and drift normalization need explicit definitions in the
  manuscript. The archived physics checks are retained for inspection.
- Results concern the reported initial state, coefficients and time interval.
  Broader initial-state or long-horizon stability studies are excluded.

The scientific manuscript text has not been revised in this source recovery.

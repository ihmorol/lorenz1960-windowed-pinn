# Archive provenance

- Public bundle source snapshot: `36ccfea89c05400aeca5d774c5ec4ad871497d09`
- Audited recovered training/results source: `babc47aab9f7cdbe8fde832cdd1e670eab1f9210`
- Archived Colab environment commit: `ed43703233264b5568ec4598dedc1cf84ac0f452`
- Archived Colab state: dirty working tree; the diff was not retained
- Run configurations: numerical authority for checkpoint reconstruction
- Checkpoint integrity: `checkpoints.sha256`
- Configuration transformations: `config-provenance.json`

The public bundle contains 16 paper-related runs. Later stability experiments
and unrelated exploratory results are excluded.

Published text files use LF line endings; numerical values and checkpoint bytes are preserved. The initial bundle contained an obsolete sequential prototype. The completed shared-network implementation, causal resume support, ablation/Colab drivers, table generator and their regression tests were recovered from the recorded ed43703 commit. The prototype was removed. The unarchived dirty diff remains unavailable, so exact historical source identity is not established. Direct evaluation of all three saved R6 checkpoints is verified.

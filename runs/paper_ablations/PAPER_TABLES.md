# Paper tables

Generated 2026-10-07 16:45 UTC from `runs/paper_ablations`.
Wall clock is per run; runs on different devices or under parallel jobs are not
timing-comparable. Seed 0 of r4_candidate is the candidate run of record.

- suite run 2026-10-07T12:38:30Z: commit ed43703233 (dirty), gpu Tesla T4, 15360 MiB, 580.82.07, jobs 4, torch 2.11.0+cu130
- suite run 2026-10-07T13:44:14Z: commit ed43703233 (dirty), gpu Tesla T4, 15360 MiB, 580.82.07, jobs 2, torch 2.11.0+cu130

## T1. Per-run results (actual optimizer work; RMSE and max error are combined L2)

| run | seed | device | params | points | Adam steps | L-BFGS evals | capped stages | RMSE | max error | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| F1_seed0 (run of record) | 0 | cuda | 11,283 | 39,793 lhs | 40,000 | -- | -- | 2.24 | 3.35 | 4529 |
| r1_no_causal_seed1 | 1 | cpu | 304,641 | 41,446 uniform | 35,100 | 33,211 | 0 | 0.000175 | 0.000737 | 1002 |
| r1_no_causal_seed2 | 2 | cpu | 304,641 | 41,446 uniform | 35,100 | 33,486 | 0 | 0.000189 | 0.000561 | 1118 |
| r1_no_causal_seed3 | 3 | cpu | 304,641 | 41,446 uniform | 35,100 | 33,385 | 0 | 0.000516 | 0.00183 | 1305 |
| r2_no_warm_seed1 | 1 | cuda | 304,641 | 41,446 uniform | 105,877 | 33,174 | 18 | 0.000176 | 0.000752 | 10501 |
| r2_no_warm_seed2 | 2 | cuda | 304,641 | 41,446 uniform | 100,185 | 33,431 | 17 | 0.000288 | 0.00117 | 9835 |
| r2_no_warm_seed3 | 3 | cuda | 304,641 | 41,446 uniform | 102,904 | 33,387 | 19 | 0.000285 | 0.000995 | 4761 |
| r3_causal_single_seed0 | 0 | cuda | 11,283 | 39,793 lhs | 40,000 | 5,529 | 4 | 0.275 | 0.677 | 8093 |
| r4_candidate_seed0 (run of record) | 0 | cuda | 304,641 | 41,446 uniform | 34,765 | 33,143 | 2 | 6.97e-05 | 0.000309 | 881 |
| r4_candidate_seed1 | 1 | cpu | 304,641 | 41,446 uniform | 33,564 | 33,163 | 2 | 8.13e-05 | 0.00023 | 1164 |
| r4_candidate_seed2 | 2 | cpu | 304,641 | 41,446 uniform | 36,759 | 33,082 | 1 | 0.00026 | 0.000502 | 1169 |
| r4_candidate_seed3 | 3 | cuda | 304,641 | 41,446 uniform | 35,996 | 33,106 | 2 | 0.000199 | 0.000612 | 3765 |
| r4_candidate_seed4 | 4 | cuda | 304,641 | 41,446 uniform | 32,757 | 32,516 | 2 | 7.5e-05 | 0.000186 | 2934 |
| r6_sequential_seed1 | 1 | cuda | 11,283 | 41,446 uniform | 35,100 | 33,759 | 0 | 0.486 | 1.23 | 4826 |
| r6_sequential_seed2 | 2 | cuda | 11,283 | 41,446 uniform | 35,100 | 33,760 | 0 | 0.648 | 1.61 | 4173 |
| r6_sequential_seed3 | 3 | cuda | 11,283 | 41,446 uniform | 35,100 | 33,403 | 0 | 0.35 | 0.892 | 2765 |

## T2. Arm summary (seed 0 runs of record included where they belong to the arm)

| arm | what changes | n | RMSE mean +/- std | RMSE range | mean Adam steps |
|---|---|---|---|---|---|
| r4_candidate | candidate config (27 windows, causal, warm start), other seeds | 5 | 0.000137 +/- 8.73e-05 | 6.97e-05 .. 0.00026 | 34,768 |
| r1_no_causal | causal weighting off; 1,300 Adam/window (35,100 total) | 3 | 0.000293 +/- 0.000193 | 0.000175 .. 0.000516 | 35,100 |
| r2_no_warm | warm start off | 3 | 0.00025 +/- 6.38e-05 | 0.000176 .. 0.000288 | 102,989 |
| r6_sequential | r1 with one network shared by all windows | 3 | 0.495 +/- 0.149 | 0.35 .. 0.648 | 35,100 |
| r3_causal_single | F1 + causal schedule, 4 x 10,000 Adam cap | 1 | 0.275 (n=1) | 0.275 .. 0.275 | 40,000 |

## T3. Paired seeds: RMSE ratio arm / comparison (>1 means the arm is worse)

| pair | common seeds | ratios | arm worse in |
|---|---|---|---|
| r1_no_causal / r4_candidate | 1, 2, 3 | 2.16, 0.726, 2.6 | 2/3 |
| r2_no_warm / r4_candidate | 1, 2, 3 | 2.17, 1.11, 1.44 | 3/3 |
| r6_sequential / r1_no_causal | 1, 2, 3 | 2.77e+03, 3.43e+03, 679 | 3/3 |
| r3_causal_single / F1 | 0 | 0.123 | 0/1 |

## T4. R7 reference-free physics (E = k^4 x^2 + l^4 y^2 drift; closure ||u(T) - u(0)||)

| run | E drift pred | E drift ref | closure pred | closure ref |
|---|---|---|---|---|
| 4x60_f64_unit_win27_causal_warm_f1cbeebd | 0.000405 | 1.76e-10 | 5.64e-05 | 1.65e-06 |
| r1_no_causal_seed1 | 0.000478 | 1.76e-10 | 0.00035 | 1.65e-06 |
| r1_no_causal_seed2 | 0.000604 | 1.76e-10 | 0.000118 | 1.65e-06 |
| r1_no_causal_seed3 | 0.000856 | 1.76e-10 | 0.00134 | 1.65e-06 |
| r2_no_warm_seed1 | 0.000707 | 1.76e-10 | 0.000386 | 1.65e-06 |
| r2_no_warm_seed2 | 0.000753 | 1.76e-10 | 0.000798 | 1.65e-06 |
| r2_no_warm_seed3 | 0.000654 | 1.76e-10 | 0.000721 | 1.65e-06 |
| r3_causal_single_seed0 | 0.0968 | 1.76e-10 | 0.665 | 1.65e-06 |
| r4_candidate_seed1 | 0.000338 | 1.76e-10 | 9.82e-05 | 1.65e-06 |
| r4_candidate_seed2 | 0.000451 | 1.76e-10 | 0.000413 | 1.65e-06 |
| r4_candidate_seed3 | 0.000442 | 1.76e-10 | 0.000541 | 1.65e-06 |
| r4_candidate_seed4 | 0.000376 | 1.76e-10 | 0.000108 | 1.65e-06 |
| r6_sequential_seed1 | 0.62 | 1.76e-10 | 0.0024 | 1.65e-06 |
| r6_sequential_seed2 | 1.43 | 1.76e-10 | 0.000392 | 1.65e-06 |
| r6_sequential_seed3 | 0.547 | 1.76e-10 | 0.0014 | 1.65e-06 |

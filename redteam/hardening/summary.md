| method | rank change | std mrr_A | std mrr_An | std hits1_A | +H mrr_A | trace var MRR | trace path P@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| lvsa_alltypes | +0.220 | 18.94 | 7.93 | 12.45 | 11.39 | 49.65 | 5.00 |
| lvsa_seed2 | +0.145 | 18.67 | 7.95 | 12.24 | 11.27 | 48.77 | 5.30 |
| lvsa | +0.000 | 18.50 | 7.87 | 12.13 | 11.17 | 47.84 | 4.90 |
| lvsa_negall | -0.157 | 18.27 | 7.23 | 11.92 | 11.07 | 47.84 | 4.90 |
| lvsa_no_logic | -0.441 | 17.86 | 6.09 | 11.67 | 10.82 | 47.84 | 4.90 |
| vsa_composition | -0.832 | 17.23 | 4.47 | 11.68 | 11.50 | 42.48 | 3.85 |
| symbolic_traversal | -3.480 | 7.15 | 3.16 | 4.28 | 6.20 | 68.54 | 3.27 |
| random | -7.853 | 0.06 | 0.04 | 0.01 | 0.07 | 0.34 | 0.00 |

| shortcut case | method | status | rank change | std mrr_A | trace var MRR |
|---|---|---|---:|---:|---:|
| shortcut_duplicate_id | shortcut_duplicate_id | failed: InvalidPrediction: test_std.npz/rank_2i: repeated entity id in row 0 | -7.883 | 0.00 | 0.00 |
| shortcut_empty | empty | partial: test_std.npz lacks rank_1p; test_std.npz lacks trace arrays; test_hplus.npz lacks rank_1p | -7.883 | 0.00 | 0.00 |
| shortcut_float_nan | shortcut_float_nan | failed: InvalidPrediction: test_std.npz/rank_3p.npy: dtype '<f8' is not an integer type | -7.883 | 0.00 | 0.00 |
| shortcut_hide_failure | broken | partial: missing test_std.npz; missing test_hplus.npz | -7.883 | 0.00 | 0.00 |
| shortcut_hide_failure | half_written | partial: missing test_hplus.npz | -1.876 | 18.50 | 47.84 |
| shortcut_hide_failure | lvsa | ok | +0.000 | 18.50 | 47.84 |
| shortcut_negative_id | shortcut_negative_id | failed: InvalidPrediction: test_std.npz/rank_2p: entity id out of range in row 5 | -7.883 | 0.00 | 0.00 |
| shortcut_object_array | object_array | failed: InvalidPrediction: test_std.npz/rank_1p.npy: dtype '|O' is not an integer type | -7.883 | 0.00 | 0.00 |
| shortcut_out_of_range_id | shortcut_out_of_range_id | failed: InvalidPrediction: test_std.npz/rank_1p: entity id out of range in row 0 | -7.883 | 0.00 | 0.00 |
| shortcut_public_data_tamper | tamper | ok | +0.000 | 18.50 | 47.84 |
| shortcut_reported_claims | claims | ok | +0.000 | 18.50 | 47.84 |
| shortcut_selective_traces | no_traces | partial: test_std.npz lacks trace arrays | -1.361 | 18.50 | 0.00 |
| shortcut_truncated_rows | shortcut_truncated_rows | failed: InvalidPrediction: test_std.npz/rank_3p: expected 2000 rows, got 1000 | -7.883 | 0.00 | 0.00 |

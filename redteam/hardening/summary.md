| method | rank change | std mrr_A | std mrr_An | std hits1_A | +H mrr_A | trace var MRR | trace path P@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| lvsa_residual_alltypes | +0.403 | 19.37 | 8.36 | 12.80 | 11.88 | 26.54 | 1.52 |
| lvsa_alltypes | +0.237 | 19.13 | 8.22 | 12.56 | 11.50 | 25.77 | 1.70 |
| lvsa_residual | +0.140 | 18.88 | 8.08 | 12.43 | 11.53 | 24.51 | 1.77 |
| lvsa | +0.000 | 18.72 | 8.07 | 12.28 | 11.29 | 23.83 | 1.43 |
| lvsa_seed2 | -0.024 | 18.80 | 8.09 | 12.42 | 11.23 | 23.01 | 1.43 |
| lvsa_no_logic | -0.588 | 17.89 | 5.75 | 11.66 | 10.74 | 23.83 | 1.43 |
| vsa_composition | -0.664 | 17.23 | 4.47 | 11.68 | 11.50 | 25.43 | 1.02 |
| symbolic_traversal | -4.596 | 7.15 | 3.16 | 4.28 | 6.20 | 0.40 | 0.00 |
| random | -7.458 | 0.06 | 0.04 | 0.01 | 0.07 | 0.20 | 0.00 |

| shortcut case | method | status | rank change | std mrr_A | trace var MRR |
|---|---|---|---:|---:|---:|
| alt_type_switch | type_switch | ok | +0.192 | 18.79 | 23.83 |
| shortcut_duplicate_id | shortcut_duplicate_id | failed: InvalidPrediction: test_std.npz/rank_2i: repeated entity id in row 0 | -7.487 | 0.00 | 0.00 |
| shortcut_empty | empty | partial: test_std.npz lacks rank_1p; test_std.npz lacks trace arrays; test_hplus.npz lacks rank_1p | -7.487 | 0.00 | 0.00 |
| shortcut_float_nan | shortcut_float_nan | failed: InvalidPrediction: test_std.npz/rank_3p.npy: dtype '<f8' is not an integer type | -7.487 | 0.00 | 0.00 |
| shortcut_hide_failure | broken | partial: missing test_std.npz; missing test_hplus.npz | -7.487 | 0.00 | 0.00 |
| shortcut_hide_failure | half_written | partial: missing test_hplus.npz | -1.938 | 18.72 | 23.83 |
| shortcut_hide_failure | lvsa | ok | +0.000 | 18.72 | 23.83 |
| shortcut_lookup_prefix | lookup_prefix | ok | -0.027 | 18.72 | 23.44 |
| shortcut_lookup_traces | lookup_traces | ok | -0.794 | 18.72 | 0.40 |
| shortcut_negative_id | shortcut_negative_id | failed: InvalidPrediction: test_std.npz/rank_2p: entity id out of range in row 5 | -7.487 | 0.00 | 0.00 |
| shortcut_object_array | object_array | failed: InvalidPrediction: test_std.npz/rank_1p.npy: dtype '|O' is not an integer type | -7.487 | 0.00 | 0.00 |
| shortcut_out_of_range_id | shortcut_out_of_range_id | failed: InvalidPrediction: test_std.npz/rank_1p: entity id out of range in row 0 | -7.487 | 0.00 | 0.00 |
| shortcut_public_data_tamper | tamper | ok | +0.000 | 18.72 | 23.83 |
| shortcut_reported_claims | claims | ok | +0.000 | 18.72 | 23.83 |
| shortcut_selective_traces | no_traces | partial: test_std.npz lacks trace arrays | -0.806 | 18.72 | 0.00 |
| shortcut_truncated_rows | shortcut_truncated_rows | failed: InvalidPrediction: test_std.npz/rank_3p: expected 2000 rows, got 1000 | -7.487 | 0.00 | 0.00 |

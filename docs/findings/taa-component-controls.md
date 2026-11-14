# Track 3 component controls

Issue #27 separates the three existing history-rejection ingredients without
changing the default algorithm or the fixed three-argument `reject_and_blend`
interface. The additional controls belong to `TAAConfig`, used by
`reconstruct_taa`.

For each valid observation, let `d` be the maximum RGB distance from history
to its neighborhood-clamped value and `m` the backward displacement magnitude.
The two retention factors are:

- Neighborhood confidence: `exp(-d / rejection_scale)`.
- Motion blend: `1 / (1 + m / motion_scale)`.

Enabled factors multiply to give the fraction of prior history weight retained.
The image clamp independently selects whether the retained history color is
clipped to the current neighborhood bounds. All methods add the actual current
group's frame count; missing observations preserve history and its weight.

| Variant | `TAAConfig` | Retention | Retained history color |
|---|---|---|---|
| Disabled control | `clamp=False, adaptive=False` | 1 | Unclipped |
| Clamp only | `adaptive=False` | 1 | Clipped |
| Blend only | `clamp=False, disagreement_weighting=False` | Motion factor | Unclipped |
| Confidence only | `clamp=False, motion_weighting=False` | Neighborhood factor | Unclipped |
| Full TAA | Defaults | Both factors | Clipped |

The existing `adaptive=False` master switch disables both retention factors.
Both new switches default to `True`, so existing configurations retain their
meaning. In particular, `clamp=False` alone still uses both retention factors;
it is not the isolated blend-only ablation.

"Confidence only" uses neighborhood confidence to discount history. Merely
displaying a confidence map would leave the reconstruction unchanged, so it
would not define a useful reconstruction ablation. Confidence is a heuristic
retention fraction, not calibrated alignment correctness. Displacement is in
pixels relative to the reference, not a measured velocity or occlusion flag.

## Reference selection

`reconstruct_taa(cube, group_size=64, config=None, *, reference="first")`
preserves the original first-group default and also accepts `reference="last"`
for comparisons using the existing baseline's last-group policy. Group
boundaries always start at frame zero. The selected existing group initializes
history once, using its actual length even when the final group is short.
All other groups are aligned to that fixed reference and processed in original
order, exactly as in `aligned_average` with the same reference argument. No
ground truth is used for reference selection, alignment, or reconstruction.

The last-reference option changes the reference grid but does not turn the
algorithm into a chronological update ending at the reference. Its last
processed update is the penultimate group. `confidence_map` and
`valid_update_mask` describe that update; for a single-group burst both are zero.
Every compared method must use the same reference, groups, crop, intensity rule,
and evaluation support. The component controls do not establish which ingredient
helps most; that conclusion requires the measured ablation table.

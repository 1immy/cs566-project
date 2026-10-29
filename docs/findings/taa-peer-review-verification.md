# Track 3 peer-review verification

Verified on October 28, 2026 against `feature/taa-history-rejection` at
`1295dd6`, after reading `AGENTS.md`, `AI-CONTEXT.md`, and `CONTRIBUTING.md`.
The original reviewed commits match the supplied review exactly: `a9f23a5` adds the
kernel, `00c0b52` adds fixed-reference burst accumulation, and `1295dd6` adds
the runner and handoff. The original suite passed 37 tests before changes.

## Findings and decisions

| Finding | Status | Independent evidence | Action and compatibility |
|---|---|---|---|
| F1 | Confirmed quality risk | Equal registered images retain 1, 1/2, 1/3 of history at offsets 0, 8, 16. Exact translation trials lose noise averaging; the real aligner also reproduces the effect. | Preserve defaults; propose a separate optional displacement-penalty switch for Track 4's future matrix. No alignment or API change. |
| F2 | Confirmed in specific regimes | Scalar counterexample reproduces. Repeated trials show substantial harm on fine structure with small groups, but benefits on uniform scenes and contaminated history. | Record both outcomes and parameter sensitivity. No noise-aware margin or retuning justified as a general improvement yet. |
| F3 | Confirmed semantics; not a defect | Existing test verifies identical confidence with/without image clipping at the same inputs. Disabled-both controls reproduce baseline coverage and weights. | Preserve semantics; clarify CLI help, test name, and experiment label `adaptive_no_clip`. Correct a separate runner tolerance failure described below. |
| F4 | Confirmed limitation; not a defect | A valid update followed by a missing final observation preserves useful history/weight but returns zero final confidence. | Add this regression and label the panel "Last retention". No aggregate field or existing output-field changes. |
| F5 | Confirmed evaluation gap; held-out quality unresolved | Exported arrays and provenance are usable, but photon probabilities and supplied RGB GT have unresolved response/time conventions. No held-out PSNR/SSIM results exist in this branch. | Make domains explicit in the manifest; leave metric protocol and preprocessing to Track 4. Synthetic MSE is characterization only. |

No blocking defect in the TAA equations was found. Neither displacement nor
last-update retention measures calibrated registration reliability. The public
three-argument interface, `TAAConfig` defaults, result fields, Track 2 source,
group partition, fixed first reference, and input data contracts are unchanged.

## Reproducible experiment design

```bash
venv/bin/python scripts/characterize_taa.py --seed 5662026 --trials 24
```

This completed **400 measurement rows**, each with 24 independent photon-noise
trials, using Python 3.14.3 and NumPy 2.5.4. Results, per-trial MSE, standard
errors, empirical squared bias/variance, parameters, seeds, and source hashes
are in ignored `results/sample_taa/peer_review/synthetic.json`.

All observations are explicitly generated Bernoulli binary frames. No SPC
ground truth or held-out data is loaded. Each comparison uses the same burst,
groups, first-group reference, warped observations, and scoring support. The
script caches warps and supplies them to the unchanged baseline and TAA
wrappers through local test patches. Static trials use identity oracle warps;
translation trials use either known exact offsets or the actual Track 2
aligner. These experiments do not replace production alignment.

Static scenes have 256 frames, 16x24 RGB pixels, mean probabilities 0.02, 0.25,
0.75, and group sizes 8, 32, 64. Scene patterns are uniform, a vertical edge,
and a one-pixel checkerboard. Edge/checker values are `p +/- min(p,1-p)/2`.
All pixels are scored. Full-TAA sensitivity varies one parameter at a time:
radius 0/1/2, margin 0/0.05/0.1, rejection scale 0.03/0.1/0.3.
Other settings retain their defaults. Naive, aligned, disabled-both,
clamp-only, adaptive-without-clipping, and full TAA are all recorded.

Seed sequences are `[5662026,0,scene_id,rate_id,trial]` for static noise,
`[5662026,1]` for fixed translation textures, `[5662026,2,trial]` for translation
noise, and `[5662026,3,trial]` for stale-history noise. Trials run from 0 to 23;
scene/rate IDs follow the order above. Static settings share each noise draw.
Translation cases share registered noise before applying their known shifts.

Standard errors use the 24 per-trial MSEs, not individual pixels as independent
trials. The empirical decomposition uses variance with `ddof=0`, so MSE equals
empirical squared bias plus variance. Finite-trial squared bias contains
sampling error and is not an unbiased estimate of population squared bias.

## F1: correct compensation still loses averaging

`align_warp` implements `warped[y,x] = mean[y+dy,x+dx]`. Positive `dx` samples
to the right; offsets refer to the fixed first group, not velocity or residual
error. Deterministic tests cover both signs and both coordinate components.

With equal registered images there is no neighborhood disagreement:

| Displacement magnitude | Retention | Weight after old=128, current=64 | Final history coefficient |
|---:|---:|---:|---:|
| 0 | 1 | 192 | 2/3 |
| 8 | 1/2 | 128 | 1/2 |
| 16 | 1/3 | 106.666667 | 0.4 |

Equal nominal weights instead give history coefficients 1/2, 1/3, 1/4.
`test_correctly_registered_displacement_still_discards_history_weight` checks
the retained fraction and accumulated weight while the image stays unchanged.

Translation probabilities are a fixed RGB texture in `[0.05,0.65)`. There are
256 frames; shifts progress by rounded linear steps from zero to the listed
last displacement. Exact-warp trials use 32x64 pixels, groups of 32, and score
`y=2:30,x=18:46`. The same interior excludes missing borders and their immediate
neighborhoods at every tested displacement. Source images are rolled to create
translations; oracle sampling never wraps missing observations.

Mean MSE (standard error in parentheses):

| Final dx | Aligned | Clamp only | Motion penalty only | Full TAA |
|---:|---:|---:|---:|---:|
| 0 | 0.000763377 (0.00000349) | 0.000834153 (0.00000446) | 0.000763377 (0.00000349) | 0.000855796 (0.00000479) |
| +/-8 | 0.000763377 (0.00000349) | 0.000834153 (0.00000446) | 0.00173834 (0.00000970) | 0.00190537 (0.0000111) |
| +/-16 | 0.000763377 (0.00000349) | 0.000834153 (0.00000446) | 0.00283418 (0.0000176) | 0.00299876 (0.0000180) |

"Motion penalty only" uses existing `clamp=False, clamp_margin=1`, which makes
neighborhood disagreement exactly zero on [0,1]; no extra production switch
is needed for this diagnostic. Its MSE is 2.277x baseline at 8 pixels and
3.713x at 16. Full TAA is 2.496x and 3.928x. The paired positive/negative
results agree exactly. Direction invariance alone is not quality evidence.

The real-aligner check uses a separate 96x96 texture, groups of 64, and common
interior `y=24:72,x=24:72`. All tested interior offsets were correct across
24 trials and 0, +/-8, +/-16 endpoints. Baseline MSE is 0.000772651;
full TAA MSE is 0.000796284, 0.001092996, and 0.001494049 at magnitudes 0, 8,
16 respectively (1.031x, 1.415x, 1.934x baseline). This does not establish
alignment accuracy for arbitrary scenes or missing borders.

A separate opt-in displacement-penalty switch is a justified proposal for
Track 4 to compare disagreement retention without conflating displacement.
It is deferred pending that experiment design; synthetic perfect registration
does not settle the tradeoff on misregistration/disocclusion. The original
method remains exactly reproducible and its default is unchanged.

## F2: clamping has both failure and beneficial regimes

For `F=0.30`, `H=0.20`, zero motion, and equal nominal weights, the measured
output is 0.28112298 with retention 0.60653067. The ordinary average is 0.25.
This deterministic counterexample is now a regression test, not a claim about
average performance under photon noise.

Selected static results, mean MSE across 24 trials:

| Scene / mean p / group size | Naive = aligned | Clamp only | Adaptive, no clipping | Full TAA |
|---|---:|---:|---:|---:|
| Uniform / 0.02 / 8 | 0.0000762297 | 0.0000694013 | 0.0000727274 | 0.0000694521 |
| Uniform / 0.25 / 64 | 0.000730477 | 0.000711310 | 0.000719835 | 0.000708603 |
| Checkerboard / 0.25 / 8 | 0.000672692 | 0.001052362 | 0.000819019 | 0.001014790 |
| Checkerboard / 0.75 / 8 | 0.000675658 | 0.001050603 | 0.000816490 | 0.000997645 |
| Checkerboard / 0.25 / 32 | 0.000672692 | 0.000649058 | 0.000666827 | 0.000661800 |

On the p=0.25 checkerboard with groups of 8, clipping increases MSE by 56.4%
and full TAA by 50.9%. The paired clamp-minus-baseline MSE difference is
0.000379670 with SE 0.0000106990. Empirical squared bias increases from
0.0000270027 to 0.000243524 and variance from 0.000645689 to 0.000808837.
Thus the loss includes both systematic spatial distortion and added noise.

Conversely, uniform p=0.25 with groups of 64 improves modestly under clamping:
paired MSE difference -0.0000191669, SE 0.000000909833. The ordinary average's
observed MSE 0.000730477 is close to Bernoulli variance `p*(1-p)/256 =
0.000732421875`. Clipping sometimes reduces variance enough to offset bias.
The review's possible-bias concern is confirmed; a blanket claim that clamping
always harms static reconstruction would be unsupported.

Sensitivity for the p=0.25 checkerboard (full TAA, one setting changed):

| Setting | MSE, groups of 8 | MSE, groups of 32 |
|---|---:|---:|
| Defaults | 0.001014790 | 0.000661800 |
| radius=0 | 0.020688984 | 0.003946159 |
| radius=2 | 0.000666065 | 0.000663757 |
| margin=0 | 0.001443671 | 0.000747701 |
| margin=0.1 | 0.000747976 | 0.000666869 |
| rejection_scale=0.03 | 0.001658834 | 0.000712788 |
| rejection_scale=0.3 | 0.001005703 | 0.000652091 |

The sweep also includes the sharp edge and group size 64. Settings that reduce
this failure are not universal optima. A noise-aware margin remains a research
proposal, not a justified default change from these limited scenes.

A favorable contaminated-history control uses two 64-frame groups, p=0.8 in
the deliberately wrong first group and p=0.2 in the current group, with the
current p=0.2 as the target. MSE falls from 0.091126892 (ordinary averaging)
to 0.004861270 (clamp only) and 0.002430920 (full TAA). This illustrates stale
history rejection; it is explicitly not a first-reference-time benchmark.

## Runner correction and regression coverage

The new sweep found a separate runner failure: a static p=0.25 Bernoulli burst
from `default_rng(5662026)`, shape `(256,16,24,3)`, grouped by 8 with identity
warps, yields a disabled-control maximum error of `1.1920928955078125e-7`.
The original `atol=1e-7, rtol=0` check raises `RuntimeError` despite exact
coverage and weight agreement. The error is from repeated float32 weighted
means versus baseline sum-then-divide, not different alignment or rejection.

The runner now records and uses `max(1e-7, (groups-1)*float32_epsilon)` as its
absolute roundoff allowance, keeping `rtol=0`. Counts and disabled weights
still require exact equality. A deterministic runner regression verifies the
valid case passes while a 0.001 error in reconstruction, counts, or weights
still fails. This is a diagnostic tolerance change; reconstruction arithmetic
is unchanged. The synthetic runner uses the same allowance.

Other additions test exact F1 weights, the F2 counterexample, invalid-neighbor
exclusion and truncated borders, final missing-update semantics, synthetic
oracle geometry, and trial statistics. The existing no-clamp test was renamed
to describe its preserved neighborhood dependence. CLI help, the image label,
and manifest domain descriptions were clarified. No aggregate-confidence
output, new dependency, or evaluation function was added.

## Sample reruns and Track 4 handoff

The following commands all completed successfully from the repository root:

```bash
venv/bin/python -m unittest discover -s tests -v
venv/bin/python scripts/run_sample_taa.py --group-size 64 --out results/sample_taa/peer_review/full
venv/bin/python scripts/run_sample_taa.py --group-size 64 --no-adaptive --out results/sample_taa/peer_review/clamp_only
venv/bin/python scripts/run_sample_taa.py --group-size 64 --no-clamp --out results/sample_taa/peer_review/adaptive_no_clip
venv/bin/python scripts/run_sample_taa.py --group-size 64 --no-clamp --no-adaptive --out results/sample_taa/peer_review/disabled
venv/bin/python scripts/run_sample_taa.py --group-size 96 --out results/sample_taa/peer_review/partial
venv/bin/python scripts/run_sample_taa.py --group-size 8 --out results/sample_taa/peer_review/small_groups
```

The final suite passed **44 tests**. All six sample configurations ran on
`kitchen1` bursts 000000/000001, frames 0:256, crop `(336,464,336,464)`.
Disabled-control maximum errors were 0 at group sizes 64 and 96 and
`8.940696716308594e-8` at group size 8 on both samples. Counts and control
weights matched exactly in every run. The 96-frame case includes a 64-frame
final group. Default final valid fractions were 0.947754/0.934204 and mean
retention over valid pixels was 0.763800/0.596340, matching the original report.

All nine exported NPZ arrays in the default runs are byte-value identical to
the pre-change runs. Source/artifact hashes in the six manifests were checked.
The comparison image was inspected; its new label states the final-update
meaning. README, webpage, source, tests, scripts, and findings were searched
for contrary confidence interpretations; none claimed calibrated probability,
temporal average, or full-burst reliability.

Track 4 has float reconstructions, the provided RGB GT, geometric frame counts,
discounted algorithmic weights, final-update validity/retention, fixed-reference
metadata, group counts, crop/frame identities, and hashes. These support a
future shared metric pipeline. `valid_update_mask` is not a cumulative quality
mask, and `history_weights` is not an effective statistical sample size. All
reference pixels have first-group evidence; `frame_counts==256` identifies
complete geometric coverage on these samples. Track 4 must choose the same
scoring support and intensity-response/reference-time conventions for every
method before reporting held-out PSNR/SSIM. Per-group warp histories and a
temporal-average retention diagnostic are not exported; they would require
explicit additional instrumentation if later needed.

No Track 2 modification is requested by these findings: the verified motion
convention is correct. The proposed displacement switch and noise-aware
clamping experiments should be agreed with Track 4 before expanding its
benchmark matrix. Quality superiority on real held-out scenes remains
unresolved.

# Contributing

## Task split
Four tracks, split along the pipeline's natural seams (see `docs/proposal.tex` Figure 1). Pick one by commenting your name next to it in the group chat, or editing this file directly and opening a PR.

Fix the interfaces below in Week 1 so everyone can build and unit-test against fixture data without waiting on anyone else's code -- integrate at the end of Weeks 2 and 4 rather than continuously.

| # | Track | Owner | Owns | Depends on |
|---|---|---|---|---|
| 1 | **Data & Infra** | Limo | Dataset download/chunking, `visionsim` synthetic motion sweeps, the burst-loading API (`load_burst(scene) -> [T,H,W] binary array`), repo/webpage upkeep | Nothing -- unblocks everyone else |
| 2 | **Alignment backbone** | | `align_warp(burst, history) -> warped_frame, motion_field` -- shared by both the baseline and the TAA pipeline; naive-averaging merge | A stub/mock burst from Track 1 (doesn't need the real loader to start) |
| 3 | **TAA history-rejection** | | `reject_and_blend(warped_frame, history, motion_field) -> merged_frame, confidence_map` -- neighborhood clamp, motion-adaptive blend weights (the novel contribution) | A stub warped frame from Track 2 |
| 4 | **Evaluation & analysis** | | `evaluate(reconstruction, ground_truth) -> PSNR, SSIM`, the motion-speed failure-analysis sweep, ablations, results plots/webpage | Stub reconstructions from Tracks 2/3 |

**Fairness notes:**
- Tracks 2 and 3 are the algorithmically heaviest; Track 3 is the paper's actual novel contribution, so consider pairing up on it rather than one person owning it solo.
- Track 1 front-loads early and frees up by Week 3 -- that person is the natural one to help Track 4 with the Week 6-8 motion-speed sweep.
- Track 4 is light early, heavy late (Weeks 6-9).

**Maps onto report sections** (so each person's writing traces to their own work):
- Track 1 -> Data section + dataset paragraphs in Related Work
- Track 2 -> Baseline half of Approach + "state of the art" framing
- Track 3 -> TAA-augmented half of Approach + "why ours should work better"
- Track 4 -> Evaluation Plan + Timeline

## Data handoff (Track 1)
The full Single Photon Challenge dataset (~425 GB) and synthetic `visionsim` sweeps live on Limo's personal NAS, not in this repo and not shared over the network to teammates for security reasons. Nobody else needs raw access to work:

- **Starting out:** a small canonical sample set (a handful of short bursts, a few hundred MB total) is committed under `data/samples/`. This matches the shape of `load_burst()`'s real output, so Tracks 2--4 can build and unit-test against it from day one without touching the full dataset.
- **Mid-sized intermediate results** (motion sweeps, alignment outputs for the mid-term report) get shared via a Google Drive folder under Limo's UW account -- link posted in the group chat -- rather than direct NAS access.
- **Final full-scale numbers** (final evaluation, ablations) are run by Limo on the NAS against the merged pipeline code from all tracks, with only the output metrics/plots shared back, not the raw data.
- `data/raw/` and other large/derived data are gitignored (see `.gitignore`); only `data/samples/` is tracked in the repo.

## Branch structure
- `main` — protected, stable milestones only (proposal submitted, mid-term submitted, final). No direct pushes; merge via PR only.
- `develop` — integration branch. All feature work merges here first.
- `gh-pages` — serves the project webpage. Push directly here when updating site content; keep it separate from code branches.

## Workflow
1. Pull latest `develop`: `git checkout develop && git pull`
2. Branch off it: `git checkout -b feature/short-description`
3. Commit small, working increments with clear messages.
4. Push your branch and open a PR into `develop`. Tag another teammate for a quick look (doesn't need to be exhaustive — just a sanity check).
5. Once merged, delete the feature branch.

## Access
I manage the repo, teammates have write access to push branches and open PRs. Branch protection on `main` means merges there go through an approval process before being integrated into main.

## Milestones (tag `main` at each)
- `proposal` — after Sep 29 submission
- `midterm` — after Oct 29 submission
- `final` — after Dec 10 webpage freeze

## Webpage updates
Edit files under `webpage/` on `main`, then sync onto `gh-pages` by copy (never merge `main` and `gh-pages` -- they have permanently different layouts, root-level on `gh-pages` vs. `webpage/`/`docs/` on `main`):
```bash
git checkout gh-pages
git show main:webpage/index.html > index.html
git show main:webpage/style.css > style.css
git show main:webpage/proposal.pdf > proposal.pdf
git add index.html style.css proposal.pdf && git commit -m "Sync webpage: [what changed]"
git push origin gh-pages
git checkout main
```

## Code style
WIP

## Questions
Post in Discord as soon as any issue comes up

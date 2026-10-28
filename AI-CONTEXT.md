# AI context: CS 566 project

Read this first if you are an AI assistant (Claude Code, Codex, Copilot, Cursor, etc.) working in this repo. `CLAUDE.md` and `AGENTS.md` point here. Humans: this is also a fair one-page summary of the project.

## 1. Project in one paragraph
Course project for CS 566 Introduction to Computer Vision (UW-Madison, Fall 2026, Prof. Mohit Gupta). Title: **Temporal Anti-Aliasing for Single-Photon Reconstruction.** SPAD cameras output bursts of binary frames, and a clean image only appears after aligning and merging a whole burst (quanta burst photography). We test whether history-rejection heuristics from real-time temporal anti-aliasing (TAA), namely **neighborhood clamping** and **motion-adaptive blending**, improve classical align-and-merge reconstruction, especially under fast motion and disocclusion. We compare a TAA-augmented pipeline against two baselines (naive averaging, hierarchical block alignment) on the Single Photon Challenge data. The contribution is an interpretable classical method with a rigorous failure analysis, not a state-of-the-art claim against neural methods. Goal for the team: earn an A. Grading is on the reports, presentation and webpage, so clear, honest, reproducible results matter more than clever code.

Links: webpage https://1immy.github.io/cs566-project/ , repo https://github.com/1immy/cs566-project , benchmark https://singlephotonchallenge.com/ , proposal source in `docs/proposal.tex`.

## 2. Team and tracks
| Track | Owner | GitHub | Owns |
|---|---|---|---|
| 1 Data & Infra | Limo Kemei | 1immy | dataset, burst loader, samples, visionsim sweeps, repo and webpage upkeep |
| 2 Alignment backbone | Ethan Shao | zshao46 | `align_warp`, naive-averaging merge, block alignment |
| 3 TAA history rejection | Adrian Gottwein | xLuckySe7en | `reject_and_blend`, clamp, adaptive blend, confidence map (the novel contribution) |
| 4 Evaluation & analysis | Jack Chen | JC637W | `evaluate`, results harness, motion-speed failure analysis, ablations, plots, report figures |

Jack may also help on Track 3 or 4. Work on the track of the person you are assisting. Do not rewrite another track's code without asking its owner; propose changes in an issue or PR comment.

## 3. Timeline (course weeks)
| Weeks | Dates | Milestone |
|---|---|---|
| 1-2 | Sep 29 - Oct 12 | Data pipeline and baseline working |
| 3-4 | Oct 13 - Oct 26 | TAA-augmented pipeline implemented |
| 5 | Oct 27 - Oct 29 | **Mid-term report due Oct 29**: baseline vs TAA |
| 6-8 | Oct 30 - Nov 19 | Motion sweeps (visionsim), failure analysis |
| 9 | Nov 20 - Nov 30 | Final evaluation, ablations |
| 10 | Dec 1 - Dec 10 | Presentation and final webpage (freeze Dec 10) |

The team is behind the Weeks 3-4 target, so the mid-term needs a working baseline, a TAA prototype and a comparison on small data, even if rough. When in doubt, prefer something that runs end to end over something polished.

## 4. Fixed interfaces (do not change signatures without agreement)
```python
load_burst(scene_dir, index=0, frames=None, crop=None) -> cube[T,H,W,3] uint8 in {0,1}   # src/burst_io.py
load_gt(scene_dir, index=0, crop=None)                 -> gt[H,W,3] float32 in [0,1]     # src/burst_io.py
load_sample(npz_path)                                  -> cube[T,H,W,3] in {0,1}         # src/burst_io.py
naive_average(cube)                                    -> [H,W,3] float32                # baseline
align_warp(burst, history)                             -> warped_frame, motion_field     # Track 2
reject_and_blend(warped_frame, history, motion_field)  -> merged_frame, confidence_map   # Track 3
evaluate(reconstruction, ground_truth)                 -> PSNR, SSIM                     # Track 4
```
Each track can build against stubs (identity warp, pass-through blend) so nobody waits. Both pipelines share the same alignment and merge backbone, so any quality difference is attributable to history rejection alone. Keep it that way: do not give the TAA pipeline an advantage elsewhere.

## 5. Data facts (verified)
- Single Photon Challenge reconstruction data: 50 training scenes with ground truth, 5 test scenes with ground truth withheld. About 425 GB uncompressed, 133 GB compressed (average train zip about 2.7 GB, not 8.5 GB).
- A scene folder contains many numbered pairs `NNNNNN.npy` + `NNNNNN.png`.
- `.npy`: `uint8`, shape `(1024, 800, 100, 3)`. That is 1024 binary frames, 800x800 px, RGB, **bit-packed along the width axis** (8 px per byte, `np.packbits` order). `np.unpackbits(a, axis=2)` gives `(1024, 800, 800, 3)`.
- `.png`: 800x800 RGB ground truth for that burst.
- Mean photon rate about 0.157 per binary pixel. A 256-frame average correlates about 0.89 with ground truth.
- A full cube unpacks to about 2 GB. Use `frames=` and `crop=` while prototyping. `crop` x bounds must be multiples of 8.
- Shared slice for teammates (Google Drive, UW-Madison accounts only): https://drive.google.com/drive/folders/1-Z96hNndg0IBuMcAzk62JWB-reFGmrEO?usp=sharing . About 5 GB, 10 scenes (`kitchen1` bursts 0, 1, 10; `library-homeoffice`, `loft`, `diner`, `staircase`, `restaurant`, `tv-couch`, `bathroom1`, `livingroom`, `simplekitchen` bursts 0, 1), same layout as the dataset.
- Motion inside a burst is real (about 2 to 20 px of mostly horizontal drift in `kitchen1`); see `docs/findings/burst-motion.md`. Alignment must run within a burst, on groups of frames, not single binary frames.
- The committed sample is `data/samples/` (two 256-frame 128x128 `kitchen1` crops, about 1 MB), made by `scripts/make_sample.py`.

## 6. Repo layout and branches
```
docs/        proposal.tex, proposal_latex.pdf, proposal.md
src/         implementation (burst_io.py so far)
scripts/     make_sample.py, setup_github.ps1
data/        gitignored except data/samples/
results/     outputs (mp4, large files ignored)
webpage/     site source on main; synced to gh-pages by copy
```
- `main`: stable milestones. Documented policy is PR only. Tag `proposal`, `midterm`, `final`.
- `develop`: integration branch. Feature branches `feature/short-description` branch off it and PR back into it.
- `gh-pages`: serves the webpage, root-level files. **Never merge `main` and `gh-pages`** (permanently different layouts). Sync by copying files with `git show main:webpage/index.html > index.html` and committing directly. See `CONTRIBUTING.md > Webpage updates`.
- Work is tracked in GitHub issues (labels `track-1-data` ... `track-4-eval`, `blocked`, `integration`, `report`; milestones follow the timeline). Reference the issue in commits and PRs (`Closes #N`).

## 7. Guardrails for AI assistants
Do not, without the owner explicitly asking:
- **Touch the NAS or ask for access to it.** The raw dataset lives on Limo's personal NAS and is deliberately not shared with teammates. Full-scale runs are done by Limo; teammates work from `data/samples/` and the Google Drive slice.
- Commit data, model weights or large files (over about 5 MB), secrets, tokens, `.env` files, or personal paths. Only `data/samples/` is tracked.
- Force-push, rewrite published history, delete branches or tags you did not create, or change branch protection.
- Merge `main` and `gh-pages`, or move webpage files out of their per-branch layout.
- Change the interface signatures in section 4, or the evaluation protocol (PSNR and SSIM on the same held-out scenes and crops for every method).
- Add AI co-author trailers (for example `Co-Authored-By: Claude ...`) or generated-by lines to commits. This repo's convention is plain commit messages under the human author's name.
- Edit the submitted proposal files under `docs/` or the PDF on `gh-pages` without asking, since they match what was handed in.

Do:
- **Never fabricate results.** Do not invent metrics, figures, citations, leaderboard numbers or baselines. Every number in a report or on the webpage must come from code in this repo that someone ran. If something has not been run, write "not yet run".
- Report negative results and failure cases plainly. The failure analysis is part of the contribution.
- Seed all randomness, record the scene and burst indices used, and keep scripts runnable from the repo root with relative paths. No hardcoded `Z:\` or home-directory paths in committed code.
- Keep changes small, tested on `data/samples/`, and scoped to the task or issue. Run it before saying it works.
- Writing style for the report and webpage: academic and plain, no em dashes, no hype. Image slots on the webpage are fixed names under `webpage/assets/`.
- Ask the human when scope, interface or data meaning is unclear. Post blockers in Discord and in the issue.

## 8. Pull requests
Open PRs into `develop` from `feature/...`, with the template in `.github/pull_request_template.md`. One logical change per PR, linked to an issue, with one teammate tagged for a quick review. PRs to `main` only for milestone releases.

## 9. Evaluation protocol (what the report will claim)
PSNR and SSIM against ground truth on held-out training scenes, for: naive averaging, block-alignment baseline, TAA-augmented pipeline. Failure analysis versus motion speed using visionsim sweeps. Ablations in Weeks 9: which TAA ingredient (clamp, adaptive blend, confidence map) helps most. Fallback if the leaderboard no longer accepts submissions: held-out training scenes only. Framing: interpretable classical method with rigorous failure analysis.

## 10. Known loose ends
- The proposal's Data section says about 8.5 GB per chunk; the real average is about 2.7 GB per zip. Fix in the final report, not the submitted proposal.
- `README.md` milestone table still shows the mid-term as "Not started".
- Code style section of `CONTRIBUTING.md` is still "WIP".
- `main` is documented as PR-only but currently accepts direct pushes from the owner.

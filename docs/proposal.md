# CS 566 Project Proposal

**Team members:** Adrian, Limo, Jack, Ethan
**Date:** [Sep 29, 2026]

## 1. Problem statement
Single-photon avalanche diode (SPAD) cameras capture high-speed bursts of binary frames: each pixel only records whether a photon arrived during an exposure, so a single frame carries almost no usable signal. Reconstructing a clean image requires aligning and merging many frames (quanta burst photography). We address **single-photon image reconstruction under motion**: specifically, whether history-rejection heuristics from real-time temporal anti-aliasing (TAA), namely neighborhood clamping and motion-adaptive blending, improve classical align-and-merge reconstruction, especially under fast motion and disocclusion.

## 2. Motivation
- **Practical importance:** SPAD arrays are an emerging sensor technology (megapixel-scale arrays now exist), which makes robust reconstruction under motion increasingly practical rather than a niche concern.
- **Scientific/technical interest:** the standard reconstruction pipeline (estimate motion, align, merge) is structurally identical to TAA in real-time rendering, a connection that, as far as we've found, hasn't been explored. Bringing game-engine history-rejection tricks into photon-level reconstruction is a genuinely cross-domain angle with a clear, testable hypothesis.
- **Fit with the course/instructor:** this builds directly on Prof. Gupta's own quanta burst photography work and the Single Photon Challenge his lab co-organized, giving us a public benchmark with ground truth rather than needing to build our own.

## 3. Related work (brief)
1. **Quanta burst photography** (Gupta's lab): establishes the align-then-accumulate pipeline for SPAD bursts that we take as our baseline.
2. **TAA survey** (NVIDIA, Yang et al. 2020): catalogs history-rejection techniques (neighborhood clamping, motion-adaptive blend weights) from real-time rendering that we adapt to this domain.
3. **Single Photon Challenge**: provides the benchmark dataset and existing leaderboard submissions as points of comparison.

## 4. Current state of the art
The current state of the art for this problem is the quanta burst photography pipeline itself: align each binary frame to a reference (via block matching or optical flow), then merge the aligned burst by averaging. This performs well under static or slowly moving scenes, but once per-frame alignment becomes unreliable, under fast motion, large disocclusion, or low light, misaligned pixels are still blended into the final estimate at full weight. The pipeline has no mechanism to detect that a given alignment is untrustworthy and downweight or discard it accordingly, which is precisely the failure mode we target.

## 5. Re-implementation vs. new approach
We plan to first **re-implement the existing quanta burst photography baseline** described above as a validated point of comparison. We then **propose a new approach** that, to our knowledge, has not been explored: adapting per-pixel history-rejection heuristics from real-time temporal anti-aliasing to explicitly detect and reject unreliable alignments, rather than blending them in regardless.

### Why we think existing approaches fall short, and why ours should work better
Existing align-and-merge pipelines have no mechanism to distinguish a reliable per-frame alignment from an unreliable one: every aligned pixel is blended in at a fixed or naively motion-scaled weight, so once alignment breaks down under fast motion or disocclusion, the error is baked directly into the reconstruction. Our TAA-augmented pipeline adds a neighborhood clamp that explicitly checks whether the aligned history agrees with the incoming frame's local neighborhood and rejects the pixel when it doesn't, plus a confidence-weighted blend that downweights marginal agreement rather than trusting it fully. Because both pipelines share the same alignment and merge backbone, any difference in reconstruction quality can be attributed specifically to this history-rejection step, isolating whether it's actually the source of improvement rather than a confound from a different underlying algorithm.

## 6. Approach
1. Implement baseline: naive frame averaging, then hierarchical block alignment + merging (simplified quanta burst photography).
2. Implement TAA-augmented pipeline: same alignment/merge backbone plus neighborhood-clamping history rejection, motion-adaptive blend weights, and a per-pixel confidence map.
3. Run both on Single Photon Challenge training scenes (held-out subset) and visionsim-generated synthetic sweeps (varying motion speed, light level).
4. Evaluate and analyze where/why each method breaks down under fast motion and disocclusion.

Course techniques used: image alignment/transformations, RANSAC, optical flow, tracking, and the quanta computational imaging lecture.

## 7. Data
**Single Photon Challenge** dataset: 50 simulated training scenes + 5 test scenes (ground truth private), each burst ("photon cube") = 1024 binary frames. Full set ~425 GB (~133 GB compressed, ~8.5 GB/chunk); plan to use 5–10 chunks, plus the small one-burst-per-scene sample already downloaded for early prototyping. Supplemented with our own synthetic sweeps via the lab's MIT-licensed **visionsim** simulator.

## 8. Evaluation plan
- PSNR/SSIM on held-out training scenes (confirming with the instructor whether the leaderboard still accepts test-set submissions).
- Failure analysis vs. motion speed using visionsim sweeps, directly answering "when does temporal reconstruction break down?"
- Framing: an interpretable classical method with rigorous failure analysis, not a state-of-the-art claim against neural approaches.

## 9. Scope and feasibility
This topic was discussed with Prof. Gupta, who confirmed it is a reasonable and worthwhile direction to propose. As expected at this stage, the specific scope may be refined as implementation progresses. He also offered to connect the team with members of his lab for data access or implementation guidance if needed.

## 10. Timeline

| Week | Dates | Milestone |
|---|---|---|
| 1–2 | Sep 29 – Oct 12 | Data pipeline (subset download, visionsim setup) + baseline (naive averaging + block alignment) working |
| 3–4 | Oct 13 – Oct 26 | TAA-augmented pipeline (clamping, adaptive blending, confidence map) implemented |
| 5 | Oct 27 – Oct 29 | Mid-term report: baseline vs. TAA results so far, difficulties |
| 6–8 | Oct 30 – Nov 19 | Motion-speed sweeps via visionsim, failure analysis, iteration |
| 9 | Nov 20 – Nov 30 | Final evaluation, ablations (which TAA tricks help most), results polish |
| 10 | Dec 1 – Dec 10 | Presentation + final webpage |

## 11. Risks / fallback plan
- **Data size:** mitigated by using a 5–10 chunk subset rather than the full 425 GB.
- **Leaderboard may not accept new submissions** (challenge has concluded): fallback is evaluation on held-out training scenes only.
- **Beating the leaderboard is unrealistic** against neural methods: reframe contribution as interpretable classical method + failure analysis, not SOTA.

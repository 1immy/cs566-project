# Anticipated questions and answers (issue #39)

Short answers for the webpage, under the video. Numbers come from `docs/findings/final-results.md`, `docs/findings/taa-study.md` and the mid-term report. Where something was not tested, the answer says so.

## Why is the ground truth at the last frame?
The ground-truth image shows the scene at the pose of the end of the burst. We found this by experiment, not from dataset documentation: aligning the groups to the first group made the result worse than naive averaging (1.5 dB worse with our fixed intensity rule, 2.2 to 2.9 dB worse with a per-image gain), and aligning to the last group made it better (1.6 dB with the fixed rule, 2.5 to 4.4 dB with a per-image gain). Our evaluation therefore aligns to the last group.

## Why does default TAA underperform plain alignment?
After alignment has compensated for the motion, a large displacement no longer means a bad pixel, but our confidence still treats it that way and discards frames that were fine. Clamping alone matches alignment (16.38 dB against 16.37 dB on `kitchen1`); the full configuration is within 0.2 dB in PSNR and about 0.04 lower in SSIM. Across 31 real bursts, TAA has no consistent PSNR difference from alignment (+0.08 dB, interval from -0.18 to +0.34) and lower SSIM on every burst (-0.073 on average). We think part of the SSIM gap is grain that TAA keeps; we did not test that.

## How does gain matching affect PSNR?
The ground truth is much brighter than the photon average, so a reconstruction has to be scaled before PSNR. The mid-term report fitted one gain per image, which flatters every method (for example, 19.7 dB for last-group alignment on `kitchen1`) and made TAA look 1.0 to 3.8 dB worse than alignment. The final evaluation uses one fixed gain of 2.7 for every image and method (16.3 dB for the same case), which is stricter, comparable across scenes, and shows TAA within 0.2 dB of alignment. We changed the gain rule and the aligner (which gained sub-pixel refinement) at the same time, and did not separate their effects, so we cannot say how much of the smaller TAA gap each one explains.

## What does the residual-based confidence change?
It was planned at mid-term: replace the displacement term with a confidence based on how well the aligned history matches the new frame, which addresses the failure above. We did not implement or evaluate it, so we have no result for it. We would expect it to stop the confidence from discarding well-aligned frames, but that is a hypothesis.

## Why are the PSNR values so low (11 to 22 dB)?
Photon bursts are very noisy: only about 16% of binary pixels fire, and a 1,024-frame average still carries visible grain. Under a fixed gain of 2.7 that suits some scenes better than others, scenes that are dark or nearly flat score lowest, and `attic` is the highest at about 22 dB. The values compare methods on the same scene, not scenes with each other.

## Do the simulated and rendered sweeps agree?
On the main points yes: naive averaging collapses within a few pixels, alignment degrades slowly, and TAA stays slightly below alignment. They disagree at one point: at 40 px of drift the simulator has TAA ahead in PSNR (23.2 against 22.4 dB) and the rendered sweep does not (14.65 against 14.80 dB). Both are one scene and one seed, so we do not read that point as a result.

## How reliable is the motion-speed measure?
Only partly. For 26 of the 31 reconstructed bursts the phase-correlation peak is below 0.05, and every burst with a good peak has a measured drift under about 19 px. The speed we report for medium and fast bursts is large measured drift, which may be real but coarsely measured, or partly noise. This is why we do not claim how TAA behaves as a function of true speed on real data.

## Did you test on the five held-out test scenes?
No. Their ground truth is not public, so every number comes from the 50 training scenes, of which we reconstructed 29.

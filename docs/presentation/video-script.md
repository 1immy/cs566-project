# Five-minute presentation: video script (issue #31)

Course limit: 5 minutes per team. This script reads in about 4:50 at a relaxed pace (145 words per minute), which leaves about 15 seconds for pauses and for the visuals to change. Every number comes from `docs/findings/final-results.md` and `docs/findings/synthetic-sweep.md`.

| Section | Speaker | Words | Read time at 145 wpm | Visual |
|---|---|---:|---:|---|
| 1. The problem | Limo | 118 | 0:49 | Slide: one binary photon frame next to a 1,024-frame average; one real burst with visible motion. |
| 2. Alignment | Ethan | 115 | 0:48 | Slide: group-of-64 diagram; the first-reference versus last-reference comparison. |
| 3. TAA method | Adrian | 121 | 0:50 | Slide: Figure 1 from the webpage (pipeline diagram). |
| 4. Results | Jack | 144 | 1:00 | Slide: per-bin table; Figure 2 (comparison grid). |
| 5. Failure analysis | Limo | 148 | 1:01 | Slide: simulated and rendered sweep table; the motion-bin caveat. |
| 6. Takeaway | Limo | 54 | 0:22 | Slide: three bullet points; the website address and repository link. |
| **Total** | | **700** | **4:50** | |

Speaker assignments follow the project tracks (alignment: Ethan, TAA: Adrian, evaluation: Jack, data, infrastructure and the failure analysis: Limo). They are a proposal and can be swapped; the text does not depend on who reads it.

## How to record your part
1. Read only your own section or sections. Record each section as its own file and leave about one second of silence at the start and end, so the sections can be joined.
2. Speak at a natural pace and time yourself with a stopwatch. If you run more than 5 seconds over the time shown, cut a sentence rather than speed up.
3. Say numbers as written, for example "zero point eight decibels". Do not add numbers that are not in the script.
4. Quiet room, same microphone distance for every take, and no music.
5. If a sentence is wrong for what you built, say so on the issue instead of changing the meaning when you read it.

## Script

## 1. The problem (Limo, about 49 s, 118 words)

*Visual:* Slide: one binary photon frame next to a 1,024-frame average; one real burst with visible motion.

A single-photon camera records, for every pixel, only whether a photon arrived. One frame is almost pure noise: only about sixteen percent of pixels fire. A usable image appears only when you merge a burst of a thousand and twenty-four frames.

But the scene moves during the burst, by several pixels and sometimes tens of pixels. So reconstruction is a loop: align the frames, then accumulate them.

That is exactly what temporal anti-aliasing does in real-time graphics, and graphics adds a trick: history rejection, which discards old pixels that no longer match. Our question: does history rejection help photon reconstruction, and when does it break? We tested it on the fifty training scenes of the Single Photon Challenge.

## 2. Alignment (Ethan, about 48 s, 115 words)

*Visual:* Slide: group-of-64 diagram; the first-reference versus last-reference comparison.

Single binary frames cannot be aligned, so we align averages of sixty-four frames, using eight-by-eight block matching on a three-level pyramid with sub-pixel refinement. Every method uses this same aligner, so differences come only from how frames are merged.

The key decision was the reference. The ground truth shows the scene at the end of the burst, which we found by trying both. Aligning to the first group is one and a half to three decibels worse than not aligning at all. Aligning to the last group is one and a half to four decibels better, depending on how the image is scaled.

So the reference frame mattered more than anything we added on top.

## 3. TAA method (Adrian, about 50 s, 121 words)

*Visual:* Slide: Figure 1 from the webpage (pipeline diagram).

For each new group, TAA first aligns it to a fixed reference group, so the new group and the running history share coordinates. It then clamps the history to the range of the surrounding three-by-three neighbourhood in the new group, which removes history that disagrees with what we now see.

It then blends the two with a confidence weight that drops when the neighbourhood disagrees, and when the pixel moved a long way.

Switch those off and we recover plain aligned averaging, so we can test each piece. Clamping alone matches alignment. What goes wrong is the displacement term: once alignment has compensated the motion, a large displacement no longer means a bad pixel, and the confidence throws away good frames.

## 4. Results (Jack, about 60 s, 144 words)

*Visual:* Slide: per-bin table; Figure 2 (comparison grid).

We evaluated thirty-one real bursts from twenty-nine scenes, grouped by how far the scene moves: slow, medium and fast. Every method gets the same aligner, the same photon data and the same fixed intensity scaling.

First: alignment clearly beats naive averaging, by about 0.8 decibels PSNR and 0.09 SSIM on average, and the PSNR gain grows from the slow to the fast bin.

Second: TAA does not beat alignment. In PSNR the average difference is under a tenth of a decibel, with an interval that includes zero. TAA wins on fourteen bursts and loses on seventeen. In SSIM it is clearer, and worse: TAA is lower on all thirty-one bursts, by about 0.07.

This grid shows the spread. On the bathroom, TAA removes alignment's block artifacts and gains two decibels. On the ultramodern scene everything is mostly noise, and TAA is two decibels worse.

## 5. Failure analysis (Limo, about 61 s, 148 words)

*Visual:* Slide: simulated and rendered sweep table; the motion-bin caveat.

To study speed, we built two sweeps with known motion: a simulator, and a Blender-rendered pan.

Naive averaging collapses almost at once: two pixels of drift cost it eight decibels. Alignment degrades slowly. TAA stays slightly below alignment at all but one of twelve points, and that one, at forty pixels in the simulator, did not appear in the rendered sweep, so we do not trust it.

On real bursts we could not show a speed at which TAA clearly fails. Part of the reason is the measurement: for twenty-six of the thirty-one bursts our motion estimate is weak, so our fast bin means large measured motion, not a known speed.

Why does TAA lose in SSIM? Our guess is that it keeps more grain, which SSIM penalises more than PSNR. We did not test that, and a confidence based on alignment residual was planned but not evaluated.

## 6. Takeaway (Limo, about 22 s, 54 words)

*Visual:* Slide: three bullet points; the website address and repository link.

Three takeaways. The reference frame matters more than any extra trick. History rejection did not beat plain alignment, and it lowered SSIM on every burst we measured. And these conclusions have limits: small samples, one intensity rule, and a motion measure we only partly trust. Code, tables and figures are on our project webpage.

## Things to keep consistent with the webpage
- The "fast" bin means large measured drift, not a known speed (26 of the 31 motion estimates are weak).
- TAA lowers SSIM on 31 of 31 bursts, and the PSNR difference has no consistent sign.
- The residual-based confidence was planned and was not evaluated.

# Motion-bin results (synthetic)

Bursts: 6. Bin edges on measured drift (px): [3.1, 9.97]. Fixed gain 2.7, same cube, reference and aligner for every method.

| bin | n | drift px | PSNR naive | PSNR aligned | PSNR TAA | SSIM naive | SSIM aligned | SSIM TAA |
|---|---|---|---|---|---|---|---|---|
| slow | 2 | 0.0 to 1.7 | 28.76 | 32.66 | 32.15 | 0.851 | 0.916 | 0.895 |
| medium | 2 | 3.8 to 7.5 | 19.20 | 30.83 | 30.01 | 0.633 | 0.901 | 0.869 |
| fast | 2 | 14.9 to 29.9 | 15.56 | 24.79 | 24.89 | 0.442 | 0.802 | 0.778 |

TAA minus aligned, mean over bursts in the bin (bootstrap 95% interval when n >= 3):

| bin | n | PSNR dB | SSIM |
|---|---|---|---|
| slow | 2 | -0.51 (n < 3, no interval) | -0.021 (n < 3, no interval) |
| medium | 2 | -0.83 (n < 3, no interval) | -0.032 (n < 3, no interval) |
| fast | 2 | +0.10 (n < 3, no interval) | -0.025 (n < 3, no interval) |

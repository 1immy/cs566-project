# Rest-of-semester milestones and issues. Safe to re-run: skips anything that already exists.
# Preview (creates nothing):   powershell -ExecutionPolicy Bypass -File scripts/setup_issues_phase2.ps1
# Create for real:             powershell -ExecutionPolicy Bypass -File scripts/setup_issues_phase2.ps1 -Apply
param([switch]$Apply)
$repo = "1immy/cs566-project"
$limo = "1immy"; $ethan = "zshao46"; $adrian = "xLuckySe7en"; $jack = "JC637W"

$haveIssues = @(gh issue list -R $repo --state all --limit 300 --json title --jq ".[].title")
$haveMiles  = @(gh api "repos/$repo/milestones?state=all" --jq ".[].title")

function Milestone($title, $due) {
  if ($haveMiles -contains $title) { Write-Host "milestone exists: $title"; return }
  Write-Host "milestone NEW:    $title ($due)"
  if ($Apply) { gh api "repos/$repo/milestones" -f title="$title" -f due_on="${due}T23:59:59Z" | Out-Null }
}

function Issue($title, $body, $label, $milestone, $assignee) {
  if ($haveIssues -contains $title) { Write-Host "issue exists:     $title"; return }
  Write-Host "issue NEW:        [$assignee] $title"
  if ($Apply) {
    gh issue create -R $repo --title $title --body $body --label $label --milestone $milestone --assignee $assignee | Out-Null
  }
}

Milestone "TAA end to end" "2026-11-08"
Milestone "Motion sweeps + failure analysis" "2026-11-19"
Milestone "Final evaluation + ablations" "2026-11-30"
Milestone "Presentation + final webpage" "2026-12-10"

$m1 = "TAA end to end"
Issue "Fix the evaluation intensity rule (gain or response curve) and document it" "Target: Oct 31. The ground truth is about 2.7x brighter than the raw photon average, so raw PSNR is misleading. Pick one rule, apply it to every method equally, and write it in docs/findings/. Every final number depends on this." "track-4-eval" $m1 $jack
Issue "Align to the last group by default in aligned_average" "Target: Nov 2. Ground truth sits at the pose of the END of the burst (see PR #16 discussion). Add a reference option, default last group, and confirm on at least one more scene." "track-2-alignment" $m1 $ethan
Issue "Subpixel offsets and per-block alignment confidence" "Target: Nov 8. Shifts are 1 to 2 px per 256 frames, so integer offsets lose accuracy. Also expose a per-block confidence so Track 3 can use it." "track-2-alignment" $m1 $ethan
Issue "TAA vs baseline on kitchen1 bursts 0, 1 and 10" "Target: Nov 8. Run naive, aligned and TAA with the agreed evaluation rule and post the table in docs/findings/." "integration" $m1 $adrian
Issue "Shared results table script that any pipeline plugs into" "Target: Nov 8. One command prints PSNR/SSIM for every method on a list of scenes and bursts. Depends on the evaluation rule issue." "track-4-eval" $m1 $jack

$m2 = "Motion sweeps + failure analysis"
Issue "Pick a scene set binned by measured motion speed" "Target: Nov 12. Use scripts/measure_burst_motion.py across scenes and bursts to bin them slow, medium and fast. Document the bins and the chosen scenes." "track-1-data" $m2 $limo
Issue "Resumable full-resolution run script for the NAS" "Target: Nov 15. Per scene and burst, writes outputs and skips finished work, so interrupted runs can continue. Only outputs are shared with the team, never raw data." "track-1-data" $m2 $limo
Issue "Failure analysis: metrics versus measured motion speed" "Target: Nov 19. PSNR/SSIM for naive, aligned and TAA across the motion bins, with plots." "track-4-eval" $m2 $jack
Issue "Failure case gallery: fast motion and disocclusion" "Target: Nov 19. Side-by-side crops where each method breaks, with the confidence map for TAA." "track-3-taa" $m2 $adrian

$m3 = "Final evaluation + ablations"
Issue "Ablations: clamp only, blend only, confidence only" "Target: Nov 27. Which TAA ingredient helps most, on the same scenes and rule as the main table." "track-3-taa" $m3 $adrian
Issue "Final full-scale runs on the NAS" "Target: Nov 28. Run the merged pipeline on the final scene set. Share metrics and plots only. Experiments freeze Nov 30." "track-1-data" $m3 $limo
Issue "Final multi-scene results table and statistics" "Target: Nov 30. Final PSNR/SSIM table with per-bin results, ready to drop into the slides and webpage." "track-4-eval" $m3 $jack
Issue "Final comparison figures: naive vs aligned vs TAA" "Target: Nov 30. Clean figures at consistent crops for slides and webpage." "track-2-alignment" $m3 $ethan

$m4 = "Presentation + final webpage"
Issue "Video script and storyboard (5 minutes)" "Target: Dec 1. About 700 words at normal pace with slide cues: problem, method, results, failure analysis, takeaway. Assign who narrates which part and time each part." "report" $m4 $limo
Issue "Record and edit the 5-minute presentation video" "Target: Dec 4 (rehearse and re-record before this). The final video is exactly the 5-minute presentation, with no Q&A in it. Check audio, legibility of figures at 1080p, and the length." "report" $m4 $limo
Issue "Upload video and embed it on the webpage" "Target: Dec 5. Host as an unlisted video, embed it in the Final Presentation section and add a plain link too (the course asks for a link to the presentation on the webpage). Add the anticipated Q&A as text under the embed. Confirm on Piazza whether a video submission is due before Dec 8." "report" $m4 $limo
Issue "Prepare Q&A answers" "Target: Dec 6. Likely questions: why the ground truth is at the last frame, why default TAA underperforms alignment, how gain matching affects PSNR, what the residual-based confidence changes. Write them as short text for the webpage." "report" $m4 $limo
Issue "Webpage: method section and diagrams" "Target: Dec 4. Limo builds the page. Adrian sends the TAA method text and a diagram draft (markdown in docs/findings/) by Dec 2." "report" $m4 $limo
Issue "Webpage: results and failure analysis text" "Target: Dec 6. Limo builds the page. Jack sends the final table, plots and a short discussion (markdown in docs/findings/) by Dec 1." "report" $m4 $limo
Issue "Webpage: data, alignment and implementation sections" "Target: Dec 6. Limo builds the page. Ethan sends the alignment write-up and final figures (markdown in docs/findings/) by Dec 2." "report" $m4 $limo
Issue "Webpage: final proof and freeze" "Target: Dec 9, freeze Dec 10. The page cannot be edited after the deadline. Check links, images, PDF and mobile layout." "report" $m4 $limo

if (-not $Apply) { Write-Host "`nPreview only. Re-run with -Apply to create the items marked NEW." }

# One-time GitHub setup: labels, milestones, starter issues.
# Needs the GitHub CLI signed in as a repo admin:  gh auth login
# Run from the repo root:  powershell -File scripts/setup_github.ps1
$repo = "1immy/cs566-project"

# GitHub usernames (all four are repo collaborators).
$who = @{ limo = "1immy"; ethan = "zshao46"; adrian = "xLuckySe7en"; jack = "JC637W" }

function Label($name, $color, $desc) { gh label create $name -R $repo --color $color --description $desc --force | Out-Null }
Label "track-1-data"       "1d76db" "Data & Infra"
Label "track-2-alignment"  "0e8a16" "Alignment backbone"
Label "track-3-taa"        "b60205" "TAA history rejection"
Label "track-4-eval"       "fbca04" "Evaluation & analysis"
Label "blocked"            "000000" "Waiting on another track"
Label "integration"        "5319e7" "Touches more than one track"
Label "report"             "c5def5" "Writing for a report or the webpage"

function Milestone($title, $due) {
  gh api "repos/$repo/milestones" -f title="$title" -f due_on="${due}T23:59:59Z" | Out-Null
}
Milestone "Mid-term report"                    "2026-10-29"
Milestone "Motion sweeps + failure analysis"   "2026-11-19"
Milestone "Final evaluation + ablations"       "2026-11-30"
Milestone "Presentation + final webpage"       "2026-12-10"

function Issue($title, $body, $label, $milestone, $assignee) {
  $a = @("issue","create","-R",$repo,"--title",$title,"--body",$body,"--label",$label,"--milestone",$milestone)
  if ($assignee) { $a += @("--assignee",$assignee) }
  gh @a
}
$mid = "Mid-term report"

# Track 1
Issue "Burst loader + committed sample (done, verify interface)" "src/burst_io.py and data/samples/ are on main. Others: confirm load_burst() returns [T,H,W,3] binary and works on the sample. Comment here if the interface needs to change." "track-1-data" $mid $who.limo
Issue "Post a shared mid-size dataset slice on Google Drive" "A few full scenes (cropped or fewer frames) for runs that need more than data/samples/. Raw data stays on the NAS." "track-1-data" $mid $who.limo
Issue "Check how much motion there is inside one 1024-frame burst" "Decides whether alignment runs within a burst or across consecutive bursts. Compare first and last 128 frames of a few bursts." "track-1-data" $mid $who.limo
Issue "visionsim synthetic motion sweeps" "Generate scenes at several motion speeds for the Weeks 6-8 failure analysis." "track-1-data" "Motion sweeps + failure analysis" $who.limo

# Track 2
Issue "align_warp(burst, history) stub with the agreed signature" "Return warped_frame, motion_field. A stub that returns the input is fine so Track 3 can start." "track-2-alignment" $mid $who.ethan
Issue "Hierarchical block alignment" "Real alignment backbone shared by the baseline and the TAA pipeline." "track-2-alignment" $mid $who.ethan
Issue "Naive-averaging merge baseline + baseline result on sample" "Reproduce the preliminary result using load_burst() and naive_average()." "track-2-alignment" $mid $who.ethan

# Track 3
Issue "reject_and_blend(warped_frame, history, motion_field) stub" "Return merged_frame, confidence_map. Pass-through stub is fine so Track 4 can wire up." "track-3-taa" $mid $who.adrian
Issue "Neighborhood clamp history rejection" "The core novel contribution. Per-pixel rejection of unreliable alignment." "track-3-taa" $mid $who.adrian
Issue "Motion-adaptive blend weights + confidence map" "Confidence-weighted blend, output the per-pixel confidence map for the report figures." "track-3-taa" $mid $who.adrian

# Track 4
Issue "evaluate(reconstruction, ground_truth) -> PSNR, SSIM" "Metric code plus tests on the sample. Report per-channel and mean." "track-4-eval" $mid $who.jack
Issue "Evaluation harness: run baseline vs TAA across scenes, write a results table" "Takes any pipeline implementing the agreed signatures." "track-4-eval" $mid $who.jack
Issue "Mid-term report draft: baseline vs TAA results and difficulties" "Due Oct 29. Pull figures and tables from the harness." "report" $mid $who.jack
Write-Host "Done. Check https://github.com/$repo/issues"

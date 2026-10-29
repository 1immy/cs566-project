# One-off: switch the existing slides/rehearsal issues to the pre-recorded video plan, then add the new video issues.
# Preview:  powershell -ExecutionPolicy Bypass -File scripts/setup_issues_video.ps1
# Apply:    powershell -ExecutionPolicy Bypass -File scripts/setup_issues_video.ps1 -Apply
param([switch]$Apply)
$repo = "1immy/cs566-project"
$renames = @{
  "Slides draft for the 5-minute talk (Dec 8)" = "Video script and storyboard (5 minutes)"
  "Two full rehearsals with timing" = "Record and edit the 5-minute presentation video"
}
$issues = gh issue list -R $repo --state all --limit 300 --json number,title | ConvertFrom-Json
foreach ($old in $renames.Keys) {
  $hit = $issues | Where-Object { $_.title -eq $old }
  if ($hit) {
    Write-Host "rename #$($hit.number): $old -> $($renames[$old])"
    if ($Apply) { gh issue edit $hit.number -R $repo --title $renames[$old] | Out-Null }
  } else { Write-Host "not found (already renamed or never created): $old" }
}
Write-Host "Now run setup_issues_phase2.ps1 -Apply to add the upload/embed and Q&A issues."
if (-not $Apply) { Write-Host "Preview only. Re-run with -Apply." }

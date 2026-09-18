# Contributing

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
Edit files under `webpage/`, then from the repo root:
```bash
git checkout gh-pages
git checkout develop -- webpage/    # pull latest webpage files from develop
git add webpage/ && git commit -m "Update webpage: [what changed]"
git push origin gh-pages
git checkout develop
```

## Code style
WIP

## Questions
Post in Discord as soon as any issue comes up

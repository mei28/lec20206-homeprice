# Retrospective

Filled by the `kaggle-retro` skill after the competition ends.

## Result

- Public / private LB and rank:
- Shake-up (public vs private per submission):

## Phases: plan vs actual

## Decisions and outcomes (closed Issues)

## What worked / what did not (with experiments as evidence)

## Top solutions vs ours

## Changes to make

- waggle (template):
- kaggle-* skills:
- nvidia-kaggle plugin notes:

## Template notes collected during the competition

- `kgl.cv.attach_folds` joins with `how="left"` but no `maintain_order`; polars 1.44 does not guarantee the left
  order by default. Outputs here were identical run to run, but the template should pass `maintain_order="left"`
  (found 2026-10-10 while writing `kgl.encoding`).

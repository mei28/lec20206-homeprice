# CLAUDE.md

Rules for agents working in a competition repository created from
[waggle](https://github.com/mei28/waggle). The `kaggle-*` skills in
`~/.claude/skills` hold the procedures (onboard, validation, experiment, submit, journal, research,
retro); this file holds what is always true here.

## Layout

| Path | Role |
|---|---|
| `input/<comp>/` | Competition data as Kaggle lays it out (gitignored) |
| `folds/` | Versioned fold files shared by all experiments (git-tracked, never overwritten) |
| `src/kgl/` | Shared modules: env, cv, metrics, features, registry, tracker, log, io |
| `experiments/expNNN_change/` | One experiment: `run.py` (train, OOF, metrics.json) and `infer.py` (submission) |
| `output/<exp>/<run>/` | Run artifacts: `metrics.json`, `config.json`, `oof.parquet`, `model/`, `run.log`, `submission.csv` |
| `notebooks/` | marimo `.py` notebooks (EDA); `.ipynb` files are export products |
| `docs/` | `competition.md`, `validation.md`, `experiments.md` (generated), `submissions.md`, `eda.md`, `retro.md`, `roadmap.md` (proposed order, decisions pending), `glossary.md`, `discussions.md` and `notebooks.md` (forum and public-kernel digests by topic) |
| `docs/explainers/`, `docs/figures/` | Picture explainers (HTML and the PNG `just explainer` renders) and generated figures (`just cv-lb-plot`) |
| `tools/` | Thin CLIs behind the justfile |

## Commands

`just --list` is the command surface. Use the recipes (`just run`, `just infer`, `just submit`,
`just exp-table`, `just record`) rather than calling Python or the Kaggle CLI directly, so every run
lands in `output/` with the same layout and the records stay complete.

## Decision gates

The user decides which experiment to run next, how validation is designed, whether a result is
accepted, and when to submit. Implement, run, record, and research without waiting, but stop at
those four points and present options unless the user has said to continue on their own.

## Phase guard

Compute the phase from the dates in `docs/competition.md`. If a request belongs to a later phase,
say which phase it belongs to and let the user decide; do not refuse.

| Phase | Time | Do | Not yet |
|---|---|---|---|
| Early | first 30% | EDA, fold design, one strong baseline, a submission pipeline that runs end to end | ensembling, TTA, heavy augmentation, hyperparameter search |
| Mid | 30-70% | 3-5 diverse single models, error analysis, data fixes, augmentation checks | full ensembling, wide hyperparameter search |
| Late | last 30% | ensembling, TTA, post-processing, final submission choice, shake-up estimate | new architectures, big preprocessing changes, new external data |

## Look at outputs before tuning

After a run, inspect at least 20 predictions against ground truth (worst and borderline cases),
group the errors, and propose changes that address a group. A proposal that only moves a
hyperparameter because "CV went down" is not enough.

## Recording

- `run.py` writes `metrics.json` the moment training finishes; never leave a finished run without it.
- After every full run: `just exp-table`, then `just record <exp> <issue>` on the idea's Issue.
- Submissions are recorded by `just submit ... --now` in `docs/submissions.md`; copy CV and LB into the
  CV vs LB table in `docs/validation.md`.
- One idea = one Issue (labels `idea`, `cv`, `data`, `bug`). Experiments are comments. Closing an Issue
  is the decision, and the user closes it.
- Three tracking Issues run for the whole competition: the diary (label `log`, pinned), the glossary
  (pinned), and CV vs LB. The `kaggle-journal` skill keeps them current in a wrap-up: after an LB is
  recorded, while a long run is going, after a research check, and at the end of work. It asks once
  with the list of what it will touch, then goes through to the push.

## Language and audience

Defaults for this repository; edit this section to change them. The skills write by it.

| Where | Language |
|---|---|
| Issue bodies and hand-written Issue comments | English, then `## 日本語訳` mirroring each heading |
| `just record` comments | English only |
| Diary comments | Japanese only |
| `docs/*.md` | Japanese prose under the template's headings |
| Explainers | ELI18: terms of art, formulas, CV numbers, and the limits of the approximation; Japanese with English sub-labels; pictures over text |

- Gloss a project abbreviation the first time it appears in a reply, and add new terms to
  `docs/glossary.md`.
- Before asking the user to decide on or close Issues, say what each one is: the problem in plain
  words, what was done, and the result with numbers.

## Code rules

- The Hard Rules of `~/.claude/AGENTS.md` apply: fail fast, no fallback code, no defensive code for
  cases that do not occur, no alternative implementation without asking.
- `src/kgl` and `tools/` follow the `tdd` skill: a failing test first, then the code. Run `just test`
  before proposing a commit.
- `experiments/` is exempt from TDD. Experiment scripts are hypotheses; their check is the CV score.
  They still fail fast: assert shapes and use `kgl.io.validate_submission`, and never swallow exceptions.
- Code that two experiments need moves into `src/kgl` with a test. Until then it stays in `run.py`.

## Experiment conventions

- Name: `exp{NNN}_{change}`. Create with `just new-exp <name> <base>`; the docstring keeps `base:`.
- Write the hypothesis in the docstring and in the Issue before writing code.
- All tunables live in the `Config` dataclass; `tyro` turns them into CLI flags
  (`--model-params.n-estimators 500`). Nested dataclasses, not dicts.
- Run ids: no overrides -> `default`; `--debug` -> `debug`; otherwise a hash of the overrides;
  `--run name` to choose. Same overrides reuse the same directory.
- Always `just run <exp> --debug` before a full run. Full runs go to the background.
- `infer.py` imports only what the Kaggle image provides plus `kgl` and its own `run.py`. It must
  produce the same `submission.csv` locally (`just infer`) and in the kernel.
- Components are named in `Config` and resolved with `kgl.registry` (`"lightgbm.LGBMClassifier"`,
  registered feature functions). Prefer adding a name over adding a branch.

## Validation

- Folds come from `folds/<name>.parquet`; `Config.folds` pins the version. A new design gets a new
  file, and `docs/validation.md` records why.
- Match the split to how the test set was built (time, group, stratified). See the
  `kaggle-validation` skill.
- Before trusting a new CV or making a first submission, run the `leak-reviewer` agent and save its
  report under `docs/reviews/`.

## Trackers and secrets

- `metrics.json` is the record. `Config.tracker` selects `none`, `wandb`, or `mlflow`
  (`kgl.tracker`); trackers are for curves, not for the record.
- Secrets stay in `.env` (gitignored) and `~/.kaggle/`. Never print `KAGGLE_API_TOKEN`, W&B keys, or
  the contents of `~/.kaggle/`.

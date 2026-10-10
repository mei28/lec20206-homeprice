set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set dotenv-load := true

default: help

# Show available recipes
help:
  @just --list

# ============================================================
# Environment
# ============================================================

# Install the uv environment, create .env from the example, fill in the Kaggle username, and write the plugin token
setup:
  @echo "Setting up the environment..."
  uv sync
  [ -f .env ] || cp .env.example .env
  u="$(uv run kaggle config view | awk '/username/ {print $NF}')"; \
    [ -n "$u" ] || { echo "Kaggle CLI is not logged in (run: kaggle auth login)"; exit 1; }; \
    sed -i "s/^KAGGLE_USERNAME=.*/KAGGLE_USERNAME=$u/" .env
  just kaggle-token
  @echo "Done. Edit COMP and SUBMIT_MODE in .env."

# Copy the current Kaggle OAuth access token into .env for the nvidia-kaggle plugin (valid a few hours; rerun on 401)
kaggle-token:
  @echo "Refreshing KAGGLE_API_TOKEN in .env..."
  @t="$(uv run kaggle auth print-access-token)"; \
    [ -n "$t" ] || { echo "no access token; run: kaggle auth login"; exit 1; }; \
    if grep -q '^KAGGLE_API_TOKEN=' .env; then sed -i "s|^KAGGLE_API_TOKEN=.*|KAGGLE_API_TOKEN=$t|" .env; else echo "KAGGLE_API_TOKEN=$t" >> .env; fi
  @echo "KAGGLE_API_TOKEN written to .env"

# Create the Issue labels used by the kaggle skills (idempotent)
labels:
  @echo "Creating labels..."
  gh label create idea --color 1d76db --description "Hypothesis to test; experiments are comments" --force
  gh label create cv --color 5319e7 --description "Validation design" --force
  gh label create data --color 0e8a16 --description "Data finding or issue" --force
  gh label create bug --color d73a4a --description "Something is broken" --force
  gh label create log --color c5def5 --description "Running diary; notes move to Issues or docs when they grow" --force

# Download competition data into input/<comp>
download comp:
  @echo "Downloading {{comp}}..."
  uv run kaggle competitions download -c {{comp}} -p input/{{comp}}
  unzip -o -q input/{{comp}}/{{comp}}.zip -d input/{{comp}}
  ls input/{{comp}}

# ============================================================
# Experiments
# ============================================================

# Build a versioned fold file: just folds skf5_seed42 --strategy StratifiedKFold --target Survived --id-col PassengerId
folds name *args:
  @echo "Building folds/{{name}}.parquet..."
  uv run python -m tools.make_folds {{name}} {{args}}

# Create experiments/<name>/ from a base experiment: just new-exp exp001_more_trees exp000_baseline
new-exp name from="exp000_baseline":
  @echo "Creating {{name}} from {{from}}..."
  uv run python -m tools.new_exp {{name}} --base {{from}}

# Train an experiment; extra args override Config fields: just run exp000_baseline --debug
run exp *args:
  @echo "Running {{exp}} {{args}}..."
  uv run python experiments/{{exp}}/run.py {{args}}

# Produce output/<exp>/<run>/submission.csv the way the Kaggle kernel will
infer exp *args:
  @echo "Inferring {{exp}} {{args}}..."
  uv run python experiments/{{exp}}/infer.py {{args}}

# Regenerate docs/experiments.md from output/*/*/metrics.json
exp-table:
  @echo "Rendering docs/experiments.md..."
  uv run python -m tools.exp_table

# Post an experiment's result to its idea Issue: just record exp000_baseline 12 [--run <run>] [--lb 0.78]
record exp issue *args:
  @echo "Commenting on #{{issue}}..."
  uv run python -m tools.issue_comment {{exp}} {{issue}} {{args}}

# ============================================================
# Submission
# ============================================================

# Validate the submission and print the submit command; add --now to submit and record the LB
submit exp msg *flags:
  @echo "Checking submission for {{exp}}..."
  uv run python -m tools.submit {{exp}} "{{msg}}" {{flags}}

# Wait for the latest submission to be scored and print the public LB
check-submission *args:
  @echo "Waiting for the latest submission..."
  uv run python -m tools.check_submission {{args}}

# ============================================================
# Records (diary, tracking Issues, explainers; the kaggle-journal skill)
# ============================================================

# Scatter of CV against the public LB per submission, colored by order (docs/figures/cv_lb.png)
cv-lb-plot:
  uv run python -m tools.cv_lb_plot

# Render an explainer page (docs/explainers/*.html) to a PNG next to it, for pasting into Issues
explainer html:
  docs/explainers/render.sh {{html}}

# ============================================================
# Code competitions (codes Dataset + model Dataset + thin notebook kernel)
# ============================================================

# Upload src/kgl, experiments/<exp>, and pyproject.toml as the <comp>-codes Dataset and wait until ready
upload-codes exp *args:
  @echo "Uploading codes for {{exp}}..."
  uv run python -m tools.upload_codes {{exp}} {{args}}

# Upload one run's model/, config.json, and metrics.json as the <comp>-<exp>-model Dataset
upload-model exp *args:
  @echo "Uploading model for {{exp}}..."
  uv run python -m tools.upload_model {{exp}} {{args}}

# Render sub/sub.py from the marimo template, export to ipynb, write kernel-metadata.json, and push the kernel
push-kernel exp *args:
  @echo "Pushing the submission kernel for {{exp}}..."
  uv run python -m tools.kernel_push {{exp}} {{args}}

# Show the status of the submission kernel's latest run
kernel-status:
  uv run kaggle kernels status "${KAGGLE_USERNAME}/${COMP}-sub"

# Print the logs of the submission kernel's latest run
kernel-logs:
  uv run kaggle kernels logs "${KAGGLE_USERNAME}/${COMP}-sub"

# ============================================================
# Notebooks
# ============================================================

# Open the EDA notebook (marimo)
eda:
  uv run marimo edit notebooks/eda.py

# Export a marimo notebook to ipynb next to it, in written order
nb-export file:
  @echo "Exporting {{file}}..."
  uv run marimo export ipynb {{file}} -o {{without_extension(file)}}.ipynb --sort top-down -f

# ============================================================
# Quality
# ============================================================

# Run the test suite (src/kgl and tools; experiments/ are not under test)
test:
  uv run pytest

# Format and fix lint findings
fmt:
  uv run ruff format .
  uv run ruff check --fix .

# Check formatting and lint without changing files
lint:
  uv run ruff format --check .
  uv run ruff check .

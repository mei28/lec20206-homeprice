"""Scatter of CV against the public LB for every recorded submission, colored by submission order.

Usage: uv run python -m tools.cv_lb_plot   (just cv-lb-plot)
Reads docs/submissions.md and writes docs/figures/cv_lb.png. The dashed line is CV = LB; a competition whose CV
is on another scale than the LB (a proxy metric, a class mix) replaces this tool in its own repository.
"""

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from tools.exp_table import parse_submissions  # noqa: E402

# ordinal blue ramp, light (oldest) to dark (newest); the lightest step still clears 2:1 on the surface
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

# label offsets in points, tried in order: right-above, right-below, left-above, left-below
SPOTS = [(6, 3), (6, -11), (-24, 3), (-24, -11)]


@dataclass(frozen=True)
class Point:
    order: int
    date: dt.date
    exp: str
    cv: float
    lb: float


def read_points(text: str) -> list[Point]:
    """One point per row of the docs/submissions.md table, numbered from 1 in the order they were submitted."""
    rows = parse_submissions(text)
    if not rows:
        raise ValueError("no submissions in the table; submit first (just submit ... --now)")
    return [
        Point(order=i, date=dt.date.fromisoformat(r["date"]), exp=r["exp"], cv=float(r["cv"]), lb=float(r["public_lb"]))
        for i, r in enumerate(rows, start=1)
    ]


def label(exp: str) -> str:
    return exp.split("_")[0].removeprefix("exp")


def place_labels(ax, labels: list[tuple[str, float, float]], marks: np.ndarray) -> None:
    """Annotate each (text, x, y) at the first spot whose box (about 22 x 9 points) overlaps no placed label and no
    marker; the last spot when none is free."""
    to_pt = ax.transData.transform
    scale = 72 / ax.figure.dpi  # display pixels to points
    marks_pt = to_pt(marks) * scale
    boxes: list[tuple[float, float, float, float]] = []

    def is_free(box: tuple[float, float, float, float]) -> bool:
        apart = all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in boxes)
        return apart and not any(box[0] - 4 < mx < box[2] + 4 and box[1] - 4 < my < box[3] + 4 for mx, my in marks_pt)

    for text, px, py in labels:
        cx, cy = to_pt((px, py)) * scale
        at = {(dx, dy): (cx + dx, cy + dy, cx + dx + 22, cy + dy + 9) for dx, dy in SPOTS}
        spot = next((s for s in SPOTS if is_free(at[s])), SPOTS[-1])
        boxes.append(at[spot])
        ax.annotate(text, (px, py), xytext=spot, textcoords="offset points", fontsize=8, color=INK_SECONDARY)


def plot(points: list[Point], out: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("order", RAMP)
    n = len(points)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    x = np.array([p.cv for p in points])
    y = np.array([p.lb for p in points])
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_SECONDARY, "text.color": INK})
    fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor=SURFACE, constrained_layout=True)
    ax.set_facecolor(SURFACE)
    span = max(x.max(), y.max()) - min(x.min(), y.min())
    pad = 0.1 * span if span > 0 else 0.01
    lo, hi = min(x.min(), y.min()) - pad, max(x.max(), y.max()) + pad
    ax.plot([lo, hi], [lo, hi], color=INK_SECONDARY, lw=1, ls="--", zorder=1)
    ax.annotate("CV = LB", (lo, lo), xytext=(8, -2), textcoords="offset points", color=INK_SECONDARY, fontsize=8)
    ax.plot(x, y, color=GRID, lw=1, zorder=1)  # the path from one submission to the next
    ax.scatter(x, y, s=70, c=colors, edgecolors=SURFACE, linewidths=2, zorder=3)
    # one label per experiment, at its latest submission
    last = {p.exp: i for i, p in enumerate(points)}
    ax.autoscale_view()
    place_labels(ax, [(label(points[i].exp), x[i], y[i]) for i in sorted(last.values())], np.column_stack([x, y]))
    ax.set_xlabel("CV", color=INK_SECONDARY)
    ax.set_ylabel("public LB", color=INK_SECONDARY)
    ax.set_title("CV vs public LB", color=INK, loc="left", fontsize=11)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY)
    bar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(1, max(n, 2))), ax=ax, shrink=0.8, pad=0.02)
    ticks = [int(t) for t in np.unique(np.linspace(1, n, min(n, 6)).round())]
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"#{t} {points[t - 1].date.strftime('%m-%d')}" for t in ticks])
    bar.set_label("submission (light = older)", color=INK_SECONDARY)
    bar.outline.set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


@dataclass
class Args:
    submissions: Path = Path("docs/submissions.md")
    out: Path = Path("docs/figures/cv_lb.png")


def main(args: Args) -> None:
    points = read_points(args.submissions.read_text())
    plot(points, args.out)
    print(f"{args.out}: {len(points)} submissions")
    for p in points:
        print(f"#{p.order} {p.date} {p.exp} cv={p.cv:.4f} lb={p.lb:.4f}")


if __name__ == "__main__":
    main(tyro.cli(Args))

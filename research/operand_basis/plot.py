"""Render saved-only operand-basis results; never loads or executes a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs/job-51df6b792d28"
REPORT_SHA = "e5d23802467bd705ef6e1d9ea7e3dd8925f52589d0dd2942c325404e171a59b7"
OUT = HERE / "figures"
BLOG_IMAGES = HERE.parents[2] / "obsidian/blog/images"
COLORS = {"source64": "#2864b7", "random64": "#e28322", "b3": "#8263a5", "zero": "#697681"}
LABELS = {"source64": "Source 64", "random64": "Random 64", "b3": "Historical B3", "zero": "Delete write"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(RUN / "report.json") == REPORT_SHA
    summary = json.loads((RUN / "advisor-independent-summary.json").read_text())
    assert summary["report_sha256"] == REPORT_SHA
    bank_path = RUN / "raw/compiled/bank_manifest.json"
    report = json.loads((RUN / "report.json").read_text())
    assert sha(bank_path) == report["raw_files"]["compiled/bank_manifest.json"]["sha256"]
    bank = json.loads(bank_path.read_text())
    OUT.mkdir(exist_ok=True)
    BLOG_IMAGES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.titleweight": "semibold"})

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.7), gridspec_kw={"width_ratios": [0.9, 1.2, 1.45]})
    energies = [bank["metrics"][key]["joint_captured_energy_fraction"] * 100
                for key in ("source_basis", "matched_random_basis")]
    axes[0].bar(["Source 64", "Random 64"], energies,
                color=[COLORS["source64"], COLORS["random64"]], width=.58)
    axes[0].set(title="Joint weight energy retained", ylabel="Folded G/U Frobenius energy (%)", ylim=(0, 15.5))
    for i, y in enumerate(energies):
        axes[0].text(i, y + .35, f"{y:.2f}%", ha="center", weight="semibold")

    arms = ["source64", "random64", "b3", "zero"]
    vals = [summary["local_by_arm"][a]["pooled_relative_l2"] for a in arms]
    xs = np.arange(4)
    axes[1].scatter(xs, vals, s=75, c=[COLORS[a] for a in arms], zorder=3)
    axes[1].axhline(1, color=COLORS["zero"], ls="--", lw=1)
    axes[1].set(title="Local write on each own history", ylabel="Pooled relative L2 error (lower is closer)",
                xticks=xs, xticklabels=["Source\n64", "Random\n64", "B3", "Zero"], ylim=(.989, 1.036))
    for x, y in zip(xs, vals):
        axes[1].annotate(f"{y:.4f}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center")
    axes[1].text(.02, .02, "Zero = 1 by construction\n273 source steps; 285 in each other arm",
                 transform=axes[1].transAxes, fontsize=8.6, color="#4b5560", va="bottom")

    pairs = summary["paired_source_contrasts"]
    comparators = ["random64", "b3", "zero"]
    x = np.arange(3)
    for i, a in enumerate(comparators):
        row = pairs[a]
        axes[2].plot([i, i], [row["source_pooled_error"], row["other_pooled_error"]], color="#b9c1c9", lw=2)
        axes[2].scatter(i-.065, row["source_pooled_error"], color=COLORS["source64"], s=65,
                        label="Source 64" if i == 0 else None, zorder=3)
        axes[2].scatter(i+.065, row["other_pooled_error"], color=COLORS[a], marker="s", s=52, zorder=3)
        axes[2].text(i, 1.025, f"Source closer at\n{row['source_lower_write_error']}/{row['same_history_pairs']} steps",
                     ha="center", va="bottom", fontsize=9)
    axes[2].axhline(1, color=COLORS["zero"], ls="--", lw=1)
    axes[2].set(title="Paired on identical consumed histories", ylabel="Pooled relative L2 error",
                xticks=x, xticklabels=["vs random", "vs B3", "vs zero"], ylim=(.989, 1.036))
    axes[2].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("More source weight energy did not produce a more faithful nonlinear write", x=.06, ha="left", fontsize=15, weight="semibold")
    fig.text(.06, .015, "Qwen2.5-1.5B-Instruct · MLP16 · three two-turn development conversations · correlated token steps, no uncertainty intervals", fontsize=9, color="#4b5560")
    fig.tight_layout(rect=(0, .06, 1, .91), w_pad=2)
    figure1 = OUT / "job-51df6b792d28-energy-and-write.png"
    fig.savefig(figure1, dpi=160)
    plt.close(fig)

    clocks = ["first", "after_answer0", "after_followup", "final"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharey=True)
    for case, ax in enumerate(axes):
        for arm in arms:
            rows = {(r["case"], r["arm"], r["clock"]): r for r in summary["readouts_candidate_vs_same_id"]}
            ys = [100*rows[(case, arm, c)]["relative_l2"] for c in clocks]
            ax.plot(np.arange(4), ys, marker="o", lw=2 if arm == "source64" else 1.4,
                    color=COLORS[arm], label=LABELS[arm], ls="--" if arm == "zero" else "-")
        ax.set(title=["Recall", "Arithmetic", "Prose"][case], xticks=np.arange(4),
               xticklabels=["First", "After\nanswer 0", "After\nfollowup", "Final"], ylim=(0, 60))
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Full-logit relative L2 error (%)")
    axes[2].legend(loc="upper right", fontsize=9, frameon=False)
    fig.suptitle("Readout fidelity changes across the conversation", x=.055, ha="left", fontsize=15, weight="semibold")
    fig.text(.055, .105, "Each line uses native replay of that arm's actual consumed IDs. Later histories can differ across arms.", fontsize=10)
    fig.text(.055, .055, "These readouts include carried cache effects. They do not isolate the current write, or score the answer just consumed.", fontsize=9.5, color="#4b5560")
    fig.tight_layout(rect=(0, .15, 1, .91), w_pad=2)
    figure2 = OUT / "job-51df6b792d28-readout-clocks.png"
    fig.savefig(figure2, dpi=160)
    plt.close(fig)
    records = []
    for path in (figure1, figure2):
        target = BLOG_IMAGES / path.name
        shutil.copy2(path, target)
        records.append({"file": path.name, "sha256": sha(path), "bytes": path.stat().st_size,
                        "blog_copy_sha256": sha(target)})
    manifest = {"schema": "tcm-operand-basis-figures-v1", "report_sha256": REPORT_SHA,
                "summary_sha256": sha(RUN / "advisor-independent-summary.json"),
                "bank_manifest_sha256": sha(bank_path), "plot_source_sha256": sha(__file__),
                "native_models_evaluated": 0, "figures": records}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

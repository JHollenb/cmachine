"""Small static scientific figures from authenticated retained arrays only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs/job-7b8a5933f1dd"
OUT = HERE / "study/figures"
OUT.mkdir(parents=True, exist_ok=True)
report = json.loads((ROOT / "report.json").read_text())
expected = "b693fc8b12dbf0e79f93a9337136517aae1d020f267467a74357ce84ed58208b"
assert hashlib.sha256((ROOT / "report.json").read_bytes()).hexdigest() == expected


def load(name):
    path = ROOT / "raw" / name
    row = report["raw_files"][name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    return np.fromfile(path, dtype="<f4").astype(np.float64)


tasks = ("recall", "math", "prose")
separation, repair, logits = [], [], []
for task in tasks:
    a, b, c = [load(f"{task}-{label}-write.f32") for label in ("source", "fiber", "repair")]
    separation.append(float(np.linalg.norm(a - b)))
    repair.append(float(np.linalg.norm(b - c)))
    z0, z1 = [load(f"{task}-{label}-s000-first.f32") for label in ("native", "fiber")]
    logits.append(100 * float(np.linalg.norm(z1 - z0) / np.linalg.norm(z0)))
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
x = np.arange(3)
axes[0].bar(x - .18, separation, .36, color="#335f9b", label="Equal-code native write separation")
axes[0].bar(x + .18, repair, .36, color="#d59132", label="Alternate write vs paid complement repair")
axes[0].set_yscale("log")
axes[0].set_ylim(3e-6, 1000)
axes[0].set_ylabel("MLP write L2 distance (log scale)")
axes[0].set_xticks(x, ("Recall", "Arithmetic", "Prose"))
axes[0].set_title("Identical 16-byte code, different native writes")
axes[0].legend(loc="upper left", fontsize=8, frameon=False)
for j, v in enumerate(separation):
    axes[0].text(j - .18, v * 1.14, f"{v:.2f}", ha="center", fontsize=9)
axes[1].bar(x, logits, .55, color="#497a67")
axes[1].set_ylabel("First full-logit difference / native logit norm (%)")
axes[1].set_xticks(x, ("Recall", "Arithmetic", "Prose"))
axes[1].set_ylim(0, 21)
axes[1].set_title("Different writes reach the same native suffix")
for j, v in enumerate(logits):
    axes[1].text(j, v + .4, f"{v:.2f}%", ha="center")
fig.savefig(OUT / "job-7b8a5933f1dd-code-collisions.png", dpi=180)
plt.close(fig)

case = next(c for c in report["driver"]["cases"] if c["task"] == "prose")
arms = {a["name"]: a for a in case["arms"]}
a, b = arms["prose-fiber"], arms["prose-native"]
grid = []
histories_a, histories_b = [], []
for j, (sa, sb) in enumerate(zip(a["steps"], b["steps"])):
    histories_a.append(sa["input_id"])
    histories_b.append(sb["input_id"])
    ka = load(f"prose-fiber-s{j:03d}-kv16-17.f32").astype("<f4").reshape(2, 2, 256)
    kb = load(f"prose-native-s{j:03d}-kv16-17.f32").astype("<f4").reshape(2, 2, 256)
    grid.append([histories_a == histories_b, ka[:, 0].tobytes() == kb[:, 0].tobytes(),
                 ka[:, 1].tobytes() == kb[:, 1].tobytes(), sa["valid_kv_sha256"] == sb["valid_kv_sha256"],
                 sa["logits_sha256"] == sb["logits_sha256"]])
matrix = np.asarray(grid, dtype=int).T
fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
ax.imshow(matrix, cmap=ListedColormap(["#bb625e", "#5c927c"]), vmin=0, vmax=1, aspect="auto")
ax.set_yticks(range(5), ("Consumed token history", "New K/V16 row", "New K/V17 row", "Complete valid K/V prefix", "Full logits"))
ax.set_xticks(range(len(grid)), ("Question", *[str(j) for j in range(1, len(grid))]))
ax.set_xlabel("Consumed transition (one-time MLP16 intervention at Question)")
ax.set_title("A matching new memory row does not restore the earlier memory")
ax.legend(handles=[Patch(color="#5c927c", label="Byte-identical / same history"), Patch(color="#bb625e", label="Different")],
          loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2, frameon=False)
fig.savefig(OUT / "job-7b8a5933f1dd-memory-clock.png", dpi=180, bbox_inches="tight")
plt.close(fig)
(OUT / "figure-data.json").write_text(json.dumps({"source_report_sha256": expected,
    "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "tasks": tasks, "native_write_separation_l2": separation, "alternate_vs_repair_l2": repair,
    "first_logits_percent_native_norm": logits, "prose_identity_matrix": matrix.tolist()}, indent=2) + "\n")
print(json.dumps({"figures": sorted(p.name for p in OUT.glob("*.png"))}))

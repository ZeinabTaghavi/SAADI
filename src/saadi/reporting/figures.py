"""Optional figures from the verified Saadi table outputs."""
import csv
from collections import defaultdict
from pathlib import Path


def plot_depth_curve(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    groups = defaultdict(list)
    with (output / "curve.csv").open() as stream:
        for row in csv.DictReader(stream):
            groups[row["dataset"]].append((int(row["cutoff"]), 100 * float(row["hr"])))
    figure, axis = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    labels = {"loogle": "LooGLE", "qasper": "QASPER", "musique": "MuSiQue", "novelhopqa": "NovelHopQA"}
    for dataset, values in groups.items():
        x, y = zip(*sorted(values))
        axis.plot(x, y, marker="o", markersize=3, label=labels.get(dataset, dataset))
    axis.set(xlabel="Retrieved chunks (k)", ylabel="HR@k (%)", xlim=(1, 20), ylim=(0, 100))
    axis.set_xticks([1, 5, 10, 15, 20])
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"curve.{extension}", dpi=180)
    plt.close(figure)

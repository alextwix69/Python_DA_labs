"""Пункт 5 ЛР № 2: сводка свидетельств для гипотезы о числе кластеров.

Запуск из корня LAB2: python "Python files/task5.py"
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "Report files"
OUTPUT = REPORT / "task5"
OUTPUT.mkdir(parents=True, exist_ok=True)
HYPOTHESES = {"iris": 3, "baseball": 3}
METHODS = [("cosine", "complete"), ("euclidean", "ward")]


def main() -> None:
    hierarchy = pd.read_csv(REPORT / "task2" / "task2_results.csv")
    variance = pd.read_csv(REPORT / "task3" / "task3_explained_variance.csv")
    projections = pd.read_csv(REPORT / "task4" / "task4_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    evidence = []
    for ax, name in zip(axes, ("iris", "baseball")):
        selected_rows = []
        for metric, method in METHODS:
            rows = hierarchy[
                (hierarchy.dataset == name)
                & (hierarchy.metric == metric)
                & (hierarchy.linkage == method)
            ].sort_values("requested_k")
            selected_rows.append(rows)
            ax.plot(rows.requested_k, rows.silhouette, "o-", label=f"{metric} + {method}")
        ax.axvline(HYPOTHESES[name], color="#a34e42", linestyle="--",
                   label=f"гипотеза k={HYPOTHESES[name]}")
        ax.set(title=name, xlabel="Число кластеров", ylabel="Коэффициент силуэта",
               xticks=range(2, 9))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        pca_2d = variance[(variance.dataset == name) & (variance.component == "PC2")].iloc[0]
        proj = projections[projections.dataset == name].iloc[0]
        for rows in selected_rows:
            for _, row in rows[rows.requested_k.isin([2, 3, 4])].iterrows():
                evidence.append({
                    "dataset": name,
                    "hypothesis_k": HYPOTHESES[name],
                    "metric": row.metric,
                    "linkage": row.linkage,
                    "k": int(row.actual_k),
                    "silhouette": float(row.silhouette),
                    "cluster_sizes": row.cluster_sizes,
                    "pca_2d_explained_variance": float(pca_2d.cumulative_variance_ratio),
                    "tsne_trustworthiness_10": float(proj.tsne_trustworthiness),
                })
    fig.suptitle("Иерархические разрезы как свидетельство для предварительной гипотезы", fontsize=14)
    fig.savefig(OUTPUT / "task5_hypotheses.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(evidence).to_csv(OUTPUT / "task5_evidence.csv", index=False, encoding="utf-8-sig")
    print("Предварительные гипотезы:", HYPOTHESES)
    print("Результаты сохранены в", OUTPUT)


if __name__ == "__main__":
    main()

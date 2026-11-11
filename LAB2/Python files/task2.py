"""Пункт 2 ЛР № 2: сравнение методов иерархической кластеризации.

Запуск из корня LAB2: python "Python files/task2.py"
Результаты: таблица CSV и сравнительные диаграммы в «Report files/task2».
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Входные"
OUTPUT = ROOT / "Report files" / "task2"
OUTPUT.mkdir(parents=True, exist_ok=True)
IRIS_FEATURES = [
    "sepal length (cm)", "sepal width (cm)",
    "petal length (cm)", "petal width (cm)",
]
BASEBALL_FEATURES = ["Height(inches)", "Weight(pounds)", "Age"]
POSITION_NAMES = {
    1: "Starting_Pitcher", 2: "Catcher", 3: "First_Baseman",
    4: "Outfielder", 5: "Relief_Pitcher", 6: "Second_Baseman",
    7: "Third_Baseman", 8: "Shortstop", 9: "Designated_Hitter",
}
METRICS = {
    "euclidean": "euclidean",
    "manhattan": "cityblock",
    "chebyshev": "chebyshev",
    "cosine": "cosine",
}
METHODS = ["single", "complete", "average", "centroid", "ward"]
SEED = 42


def load_datasets() -> dict[str, np.ndarray]:
    iris = pd.read_csv(
        INPUT / "iris.csv", skiprows=1, header=None,
        names=[*IRIS_FEATURES, "species"],
    )
    baseball = pd.read_csv(INPUT / "baseball.csv")
    if iris[IRIS_FEATURES].isna().any().any() or baseball[[*BASEBALL_FEATURES, "Position"]].isna().any().any():
        raise ValueError("Исходные данные содержат пропуски")

    # Position — категория. One-hot сохраняет различие категорий без ложного
    # порядка между исходными числовыми кодами 1..9.
    numeric = StandardScaler().fit_transform(baseball[BASEBALL_FEATURES])
    position_names = baseball["Position"].astype(int).map(POSITION_NAMES)
    if position_names.isna().any():
        raise ValueError("В baseball.csv есть неизвестные коды Position")
    position = pd.get_dummies(position_names, prefix="Position").to_numpy(dtype=float)
    baseball_matrix = np.column_stack([numeric, position])
    return {
        "iris": StandardScaler().fit_transform(iris[IRIS_FEATURES]),
        "baseball": baseball_matrix,
    }


def cluster_sizes(labels: np.ndarray) -> str:
    return ", ".join(str(x) for x in sorted(np.unique(labels, return_counts=True)[1], reverse=True))


def analyse_dataset(name: str, matrix: np.ndarray) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rows = []
    trees = {}
    n = len(matrix)
    sample = (
        np.arange(n) if n <= 500
        else np.sort(np.random.default_rng(SEED).choice(n, size=500, replace=False))
    )
    for metric_name, scipy_metric in METRICS.items():
        condensed = pdist(matrix, metric=scipy_metric)
        sampled_distances = squareform(condensed)[np.ix_(sample, sample)]
        for method in METHODS:
            # Центроидный метод и метод Уорда определены здесь только для
            # евклидовой геометрии. Другие сочетания сознательно исключены.
            if method in {"centroid", "ward"} and metric_name != "euclidean":
                continue
            tree = linkage(condensed, method=method, optimal_ordering=False)
            trees[(metric_name, method)] = tree
            cophenetic_r = float(cophenet(tree, condensed)[0])
            for requested_k in range(2, 9):
                labels = fcluster(tree, t=requested_k, criterion="maxclust")
                actual_k = len(np.unique(labels))
                sampled_labels = labels[sample]
                sampled_k = len(np.unique(sampled_labels))
                if 2 <= sampled_k < len(sample):
                    silhouette = float(silhouette_score(sampled_distances, sampled_labels, metric="precomputed"))
                else:
                    silhouette = float("nan")
                sizes = np.unique(labels, return_counts=True)[1]
                # Высота слияния, превращающего k кластеров в k-1.
                cut_height = float(tree[-(actual_k - 1), 2]) if actual_k > 1 else float("nan")
                rows.append({
                    "dataset": name,
                    "metric": metric_name,
                    "linkage": method,
                    "requested_k": requested_k,
                    "actual_k": actual_k,
                    "silhouette": silhouette,
                    "cophenetic_r": cophenetic_r,
                    "min_cluster_size": int(sizes.min()),
                    "cluster_sizes": cluster_sizes(labels),
                    "next_merge_distance": cut_height,
                })
    return pd.DataFrame(rows), trees


def select_examples(results: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series]:
    # Среди результатов с группами >= 5% выбираем наиболее отчётливое
    # разделение по silhouette. Для сравнения берём наименьшее silhouette.
    eligible = results[results["min_cluster_size"] >= int(np.ceil(0.05 * n))]
    if eligible.empty:
        eligible = results
    best = eligible.sort_values(["silhouette", "cophenetic_r"], ascending=False).iloc[0]
    worst = results.sort_values("silhouette", ascending=True).iloc[0]
    return best, worst


def draw_case(ax_tree, ax_dist, tree: np.ndarray, row: pd.Series, title: str) -> None:
    k = int(row["actual_k"])
    dendrogram(
        tree, ax=ax_tree, truncate_mode="lastp", p=25,
        no_labels=True, show_leaf_counts=False,
        color_threshold=0, above_threshold_color="#406d91",
    )
    ax_tree.set(title=f"{title}: {row['metric']} + {row['linkage']}, k={k}", ylabel="Расстояние слияния")
    ax_tree.grid(axis="y", alpha=0.25)

    # Показываем последние слияния: по оси X — число кластеров после слияния.
    remaining = np.arange(1, min(20, len(tree)) + 1)
    heights = tree[-len(remaining):, 2][::-1]
    ax_dist.plot(remaining, heights, "o-", ms=3, color="#c3583d")
    ax_dist.axvline(k, color="#2f7d64", linestyle="--", label=f"k={k}")
    ax_dist.set(
        title="Как меняется расстояние при объединении",
        xlabel="Число кластеров после слияния",
        ylabel="Расстояние слияния",
        xlim=(1, len(remaining)),
    )
    ax_dist.set_xticks([x for x in (1, 2, 3, 5, 10, 15, 20) if x <= len(remaining)])
    ax_dist.grid(alpha=0.25)
    ax_dist.legend()


def save_comparison(name: str, results: pd.DataFrame, trees: dict[tuple[str, str], np.ndarray], n: int) -> None:
    best, worst = select_examples(results, n)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    draw_case(axes[0, 0], axes[0, 1], trees[(best["metric"], best["linkage"])], best, "Удачный пример")
    draw_case(axes[1, 0], axes[1, 1], trees[(worst["metric"], worst["linkage"])], worst, "Менее удачный пример")
    fig.suptitle(f"{name}: сравнение иерархических кластеризаций", fontsize=16)
    fig.savefig(OUTPUT / f"task2_{name}_comparison.png", dpi=170)
    plt.close(fig)
    for label, row in (("best", best), ("worst", worst)):
        print(
            f"{name} {label}: {row['metric']} + {row['linkage']}, "
            f"k={row['actual_k']}, silhouette={row['silhouette']:.3f}, "
            f"sizes={row['cluster_sizes']}, cophenetic_r={row['cophenetic_r']:.3f}"
        )


def main() -> None:
    all_results = []
    for name, matrix in load_datasets().items():
        results, trees = analyse_dataset(name, matrix)
        all_results.append(results)
        save_comparison(name, results, trees, len(matrix))
    table = pd.concat(all_results, ignore_index=True)
    table.to_csv(OUTPUT / "task2_results.csv", index=False, encoding="utf-8-sig")
    print(f"Сохранены {len(table)} строк результатов и две сравнительные диаграммы в {OUTPUT}")


if __name__ == "__main__":
    main()

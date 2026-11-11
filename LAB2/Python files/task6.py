"""Пункт 6 ЛР № 2: проверка гипотезы k=3 методом k-средних.

Запуск из корня LAB2: python "Python files/task6.py"
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Входные"
OUTPUT = ROOT / "Report files" / "task6"
OUTPUT.mkdir(parents=True, exist_ok=True)
IRIS_FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
BASEBALL_FEATURES = ["Height(inches)", "Weight(pounds)", "Age"]
POSITION_NAMES = {
    1: "Starting_Pitcher", 2: "Catcher", 3: "First_Baseman",
    4: "Outfielder", 5: "Relief_Pitcher", 6: "Second_Baseman",
    7: "Third_Baseman", 8: "Shortstop", 9: "Designated_Hitter",
}
SEED = 42
N_INIT = 30
MAX_ITER = 500
HYPOTHESIS_K = 3


def load_datasets() -> dict[str, pd.DataFrame]:
    iris = pd.read_csv(INPUT / "iris.csv", skiprows=1, header=None,
                       names=[*IRIS_FEATURES, "species"])
    baseball = pd.read_csv(INPUT / "baseball.csv")
    if iris[IRIS_FEATURES].isna().any().any() or baseball[[*BASEBALL_FEATURES, "Position"]].isna().any().any():
        raise ValueError("Исходные данные содержат пропуски")
    position_names = baseball["Position"].astype(int).map(POSITION_NAMES)
    if position_names.isna().any():
        raise ValueError("В baseball.csv есть неизвестные коды Position")
    iris_scaled = pd.DataFrame(StandardScaler().fit_transform(iris[IRIS_FEATURES]), columns=IRIS_FEATURES)
    baseball_scaled = pd.DataFrame(StandardScaler().fit_transform(baseball[BASEBALL_FEATURES]),
                                   columns=BASEBALL_FEATURES)
    return {
        "iris": iris_scaled,
        "baseball": pd.concat([baseball_scaled,
                               pd.get_dummies(position_names, prefix="Position", dtype=float)], axis=1),
    }


def fit_kmeans(data: pd.DataFrame, k: int) -> KMeans:
    return KMeans(n_clusters=k, random_state=SEED, n_init=N_INIT,
                  max_iter=MAX_ITER, algorithm="lloyd").fit(data)


def save_diagnostics(name: str, table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(table.k, table.inertia, "o-", color="#327498")
    axes[0].set(title="Сумма квадратов внутри кластеров (inertia)", xlabel="k", ylabel="Inertia")
    valid = table[table.k >= 2]
    axes[1].plot(valid.k, valid.silhouette, "o-", color="#b85c3d")
    axes[1].set(title="Коэффициент силуэта", xlabel="k", ylabel="Silhouette")
    for ax in axes:
        ax.axvline(HYPOTHESIS_K, ls="--", color="#2f7d64", label="гипотеза k=3")
        ax.set_xticks(range(1, 9))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"k-means: {name}; сравнение k=1…8", fontsize=14)
    fig.savefig(OUTPUT / f"task6_{name}_diagnostics.png", dpi=180)
    plt.close(fig)


def analyse(name: str, data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in range(1, 9):
        model = fit_kmeans(data, k)
        labels = model.labels_
        sizes = np.bincount(labels, minlength=k)
        if k >= 2:
            sample_size = min(len(data), 500)
            silhouette = float(silhouette_score(data, labels, sample_size=sample_size,
                                                random_state=SEED))
            calinski = float(calinski_harabasz_score(data, labels))
            davies = float(davies_bouldin_score(data, labels))
        else:
            silhouette = calinski = davies = float("nan")
        rows.append({
            "dataset": name, "k": k, "inertia": float(model.inertia_),
            "silhouette": silhouette, "calinski_harabasz": calinski,
            "davies_bouldin": davies,
            "cluster_sizes": ", ".join(map(str, sorted(sizes, reverse=True))),
            "min_cluster_size": int(sizes.min()),
            "seed": SEED, "n_init": N_INIT, "max_iter": MAX_ITER,
        })
        if k == HYPOTHESIS_K:
            pd.DataFrame({
                "row_id": np.arange(len(data)), "cluster": labels + 1,
            }).to_csv(OUTPUT / f"task6_{name}_labels.csv", index=False, encoding="utf-8-sig")
    table = pd.DataFrame(rows)
    save_diagnostics(name, table)
    print(f"{name}:\n{table[['k', 'inertia', 'silhouette', 'cluster_sizes']].round(3).to_string(index=False)}")
    return table


def main() -> None:
    results = [analyse(name, data) for name, data in load_datasets().items()]
    pd.concat(results, ignore_index=True).to_csv(OUTPUT / "task6_results.csv",
                                                  index=False, encoding="utf-8-sig")
    print("Результаты сохранены в", OUTPUT)


if __name__ == "__main__":
    main()

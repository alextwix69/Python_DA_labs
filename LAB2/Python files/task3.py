"""Пункт 3 ЛР № 2: PCA для Iris и бейсболистов.

Запуск из корня LAB2: python "Python files/task3.py"
Результаты сохраняются в папке «Report files/task3».
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Входные"
OUTPUT = ROOT / "Report files" / "task3"
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


def load_datasets() -> dict[str, pd.DataFrame]:
    iris = pd.read_csv(
        INPUT / "iris.csv", skiprows=1, header=None,
        names=[*IRIS_FEATURES, "species"],
    )
    baseball = pd.read_csv(INPUT / "baseball.csv")
    if iris[IRIS_FEATURES].isna().any().any() or baseball[[*BASEBALL_FEATURES, "Position"]].isna().any().any():
        raise ValueError("Исходные данные содержат пропуски")
    position_names = baseball["Position"].astype(int).map(POSITION_NAMES)
    if position_names.isna().any():
        raise ValueError("В baseball.csv есть неизвестные коды Position")

    iris_numeric = pd.DataFrame(
        StandardScaler().fit_transform(iris[IRIS_FEATURES]), columns=IRIS_FEATURES,
    )
    baseball_numeric = pd.DataFrame(
        StandardScaler().fit_transform(baseball[BASEBALL_FEATURES]),
        columns=BASEBALL_FEATURES,
    )
    baseball_position = pd.get_dummies(position_names, prefix="Position", dtype=float)
    baseball_matrix = pd.concat([baseball_numeric, baseball_position], axis=1)
    return {"iris": iris_numeric, "baseball": baseball_matrix}


def save_dashboard(name: str, scores: np.ndarray, variance: np.ndarray, loadings: pd.DataFrame) -> None:
    cumulative = np.cumsum(variance)
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1], height_ratios=[1, 1.15])
    ax_scatter = fig.add_subplot(grid[:, 0])
    ax_variance = fig.add_subplot(grid[0, 1])
    ax_loadings = fig.add_subplot(grid[1, 1])

    ax_scatter.scatter(scores[:, 0], scores[:, 1], s=28 if name == "iris" else 12,
                       alpha=0.68 if name == "iris" else 0.42, color="#327498", edgecolors="none")
    ax_scatter.axhline(0, color="gray", lw=0.7)
    ax_scatter.axvline(0, color="gray", lw=0.7)
    ax_scatter.set(
        title="Проекция наблюдений на плоскость PC1–PC2",
        xlabel=f"PC1 — {variance[0]:.1%} дисперсии",
        ylabel=f"PC2 — {variance[1]:.1%} дисперсии",
    )
    ax_scatter.grid(alpha=0.2)

    component_numbers = np.arange(1, len(variance) + 1)
    ax_variance.bar(component_numbers, variance, color="#77a7bd", label="Отдельная компонента")
    ax_variance.plot(component_numbers, cumulative, "o-", color="#bb5f39", label="Накопленная доля")
    ax_variance.axhline(0.9, color="gray", ls="--", lw=1, label="90%")
    ax_variance.set(
        title="Объяснённая дисперсия",
        xlabel="Номер главной компоненты", ylabel="Доля дисперсии", ylim=(0, 1.05),
        xticks=component_numbers,
    )
    ax_variance.legend(fontsize=9)
    ax_variance.grid(axis="y", alpha=0.2)

    weights = loadings[["PC1", "PC2"]].to_numpy()
    image = ax_loadings.imshow(weights, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax_loadings.set(
        title="Нагрузки исходных признаков на PC1 и PC2",
        xticks=[0, 1], xticklabels=["PC1", "PC2"],
        yticks=np.arange(len(loadings)), yticklabels=loadings.index,
    )
    for i in range(len(loadings)):
        for j in range(2):
            ax_loadings.text(j, i, f"{weights[i, j]:+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax_loadings, label="Коэффициент компоненты", shrink=0.85)
    fig.suptitle(f"PCA: {name}; первые две компоненты сохраняют {cumulative[1]:.1%} дисперсии", fontsize=15)
    fig.savefig(OUTPUT / f"task3_{name}_pca.png", dpi=170)
    plt.close(fig)


def analyse(name: str, data: pd.DataFrame) -> pd.DataFrame:
    matrix = data.to_numpy(dtype=float)
    pca = PCA(n_components=min(matrix.shape), svd_solver="full")
    scores = pca.fit_transform(matrix)
    component_names = [f"PC{i}" for i in range(1, len(pca.explained_variance_ratio_) + 1)]
    loadings = pd.DataFrame(pca.components_.T, index=data.columns, columns=component_names)
    loadings.to_csv(OUTPUT / f"task3_{name}_loadings.csv", encoding="utf-8-sig", index_label="feature")
    variance = pca.explained_variance_ratio_
    cumulative = np.cumsum(variance)
    variance_table = pd.DataFrame({
        "dataset": name,
        "component": component_names,
        "explained_variance_ratio": variance,
        "cumulative_variance_ratio": cumulative,
    })
    save_dashboard(name, scores, variance, loadings)

    # Проверка качества двумерного приближения в исходном подготовленном
    # пространстве; результат равен накопленной объяснённой дисперсии PC1+PC2.
    two_component_reconstruction = scores[:, :2] @ pca.components_[:2] + pca.mean_
    total_squared_error = np.square(matrix - matrix.mean(axis=0)).sum()
    reconstruction_r2 = 1 - np.square(matrix - two_component_reconstruction).sum() / total_squared_error
    thresholds = {t: int(np.searchsorted(cumulative, t) + 1) for t in (0.8, 0.9, 0.95)}
    print(
        f"{name}: n={len(data)}, features={data.shape[1]}, "
        f"PC1={variance[0]:.3f}, PC2={variance[1]:.3f}, "
        f"PC1+PC2={cumulative[1]:.3f}, reconstruction_R2={reconstruction_r2:.3f}, "
        f"components_for_80/90/95%={thresholds[0.8]}/{thresholds[0.9]}/{thresholds[0.95]}"
    )
    print(loadings[["PC1", "PC2"]].round(3).to_string())
    return variance_table


def main() -> None:
    tables = [analyse(name, data) for name, data in load_datasets().items()]
    pd.concat(tables, ignore_index=True).to_csv(
        OUTPUT / "task3_explained_variance.csv", index=False, encoding="utf-8-sig",
    )
    print("Результаты сохранены в", OUTPUT)


if __name__ == "__main__":
    main()

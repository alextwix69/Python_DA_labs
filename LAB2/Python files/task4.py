"""Пункт 4 ЛР № 2: t-SNE и сравнение с двумерной PCA.

Запуск из корня LAB2: python "Python files/task4.py"
Результаты сохраняются в папке «Report files/task4».
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Входные"
OUTPUT = ROOT / "Report files" / "task4"
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
SEED = 42
PERPLEXITY = 30
MAX_ITER = 1500
NEIGHBORS_FOR_QUALITY = 10


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

    iris_scaled = pd.DataFrame(
        StandardScaler().fit_transform(iris[IRIS_FEATURES]), columns=IRIS_FEATURES,
    )
    baseball_scaled = pd.DataFrame(
        StandardScaler().fit_transform(baseball[BASEBALL_FEATURES]),
        columns=BASEBALL_FEATURES,
    )
    baseball_position = pd.get_dummies(position_names, prefix="Position", dtype=float)
    return {
        "iris": iris_scaled,
        "baseball": pd.concat([baseball_scaled, baseball_position], axis=1),
    }


def save_comparison(name: str, pca_xy: np.ndarray, tsne_xy: np.ndarray,
                    pca_variance: float, pca_quality: float, tsne_quality: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    point_size = 24 if name == "iris" else 10
    alpha = 0.7 if name == "iris" else 0.42
    for ax, xy in zip(axes, (pca_xy, tsne_xy)):
        ax.scatter(xy[:, 0], xy[:, 1], s=point_size, alpha=alpha,
                   color="#327498", edgecolors="none")
        ax.grid(alpha=0.2)
    axes[0].set(
        title=f"PCA: {pca_variance:.1%} дисперсии; сохранение соседств {pca_quality:.3f}",
        xlabel="PC1", ylabel="PC2",
    )
    axes[1].set(
        title=f"t-SNE: сохранение соседств {tsne_quality:.3f}",
        xlabel="t-SNE 1", ylabel="t-SNE 2",
    )
    fig.suptitle(f"{name}: проекции одних и тех же наблюдений без кластерных меток", fontsize=15)
    fig.savefig(OUTPUT / f"task4_{name}_tsne_vs_pca.png", dpi=180)
    plt.close(fig)


def analyse(name: str, data: pd.DataFrame) -> dict[str, float | str | int]:
    matrix = data.to_numpy(dtype=float)
    pca = PCA(n_components=2, svd_solver="full")
    pca_xy = pca.fit_transform(matrix)
    tsne = TSNE(
        n_components=2, perplexity=PERPLEXITY, learning_rate="auto",
        init="pca", max_iter=MAX_ITER, random_state=SEED,
        metric="euclidean", method="barnes_hut",
        early_exaggeration=12.0, angle=0.5,
    )
    tsne_xy = tsne.fit_transform(matrix)
    pca_quality = float(trustworthiness(matrix, pca_xy, n_neighbors=NEIGHBORS_FOR_QUALITY))
    tsne_quality = float(trustworthiness(matrix, tsne_xy, n_neighbors=NEIGHBORS_FOR_QUALITY))
    pca_variance = float(pca.explained_variance_ratio_.sum())

    coordinates = pd.DataFrame({
        "PC1": pca_xy[:, 0], "PC2": pca_xy[:, 1],
        "TSNE1": tsne_xy[:, 0], "TSNE2": tsne_xy[:, 1],
    })
    coordinates.to_csv(OUTPUT / f"task4_{name}_coordinates.csv", index=False, encoding="utf-8-sig")
    save_comparison(name, pca_xy, tsne_xy, pca_variance, pca_quality, tsne_quality)
    print(
        f"{name}: n={len(data)}, PCA variance={pca_variance:.3f}, "
        f"trustworthiness PCA={pca_quality:.3f}, t-SNE={tsne_quality:.3f}, "
        f"t-SNE KL={tsne.kl_divergence_:.3f}"
    )
    return {
        "dataset": name,
        "n_samples": len(data),
        "perplexity": PERPLEXITY,
        "learning_rate": "auto",
        "max_iter": MAX_ITER,
        "init": "pca",
        "metric": "euclidean",
        "method": "barnes_hut",
        "early_exaggeration": 12.0,
        "angle": 0.5,
        "seed": SEED,
        "quality_neighbors": NEIGHBORS_FOR_QUALITY,
        "pca_explained_variance_2d": pca_variance,
        "pca_trustworthiness": pca_quality,
        "tsne_trustworthiness": tsne_quality,
        "tsne_kl_divergence": float(tsne.kl_divergence_),
    }


def main() -> None:
    rows = [analyse(name, data) for name, data in load_datasets().items()]
    pd.DataFrame(rows).to_csv(OUTPUT / "task4_summary.csv", index=False, encoding="utf-8-sig")
    print("Результаты сохранены в", OUTPUT)


if __name__ == "__main__":
    main()

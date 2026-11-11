"""Пункт 7 ЛР № 2: исходные 3D-графики и распределение k-means.

Запуск из корня LAB2: python "Python files/task7.py"
Перед запуском требуется выполнить task6.py для получения меток k=3.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Входные"
REPORT = ROOT / "Report files"
OUTPUT = REPORT / "task7"
OUTPUT.mkdir(parents=True, exist_ok=True)
IRIS_FEATURES = ["sepal length (cm)", "sepal width (cm)",
                 "petal length (cm)", "petal width (cm)"]
IRIS_AXES = [IRIS_FEATURES[0], IRIS_FEATURES[1], IRIS_FEATURES[3]]
BASEBALL_AXES = ["Height(inches)", "Weight(pounds)", "Age"]
POSITION_NAMES = {
    1: "Starting_Pitcher", 2: "Catcher", 3: "First_Baseman",
    4: "Outfielder", 5: "Relief_Pitcher", 6: "Second_Baseman",
    7: "Third_Baseman", 8: "Shortstop", 9: "Designated_Hitter",
}


def load_labels(name: str, n: int) -> np.ndarray:
    labels = pd.read_csv(REPORT / "task6" / f"task6_{name}_labels.csv")
    if len(labels) != n or not np.array_equal(labels.row_id.to_numpy(), np.arange(n)):
        raise ValueError(f"Метки {name} не соответствуют порядку исходных наблюдений")
    return labels.cluster.to_numpy(dtype=int)


def draw_3d(name: str, data: pd.DataFrame, features: list[str], labels: np.ndarray,
            original: bool) -> None:
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    position = data["Position"].astype(int) if name == "baseball" else None
    if original and position is not None:
        palette = plt.get_cmap("tab10")
        colors = [palette((int(code) - 1) % 10) for code in position]
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=palette((code - 1) % 10),
                   label=f"{code}: {POSITION_NAMES[code]}")
            for code in sorted(position.unique())
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        subtitle = "цвет — Position, исходная категория"
    elif original:
        colors = "#447996"
        subtitle = "без меток кластеров"
    else:
        palette = plt.get_cmap("tab10")
        colors = [palette(int(label) - 1) for label in labels]
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=palette(cluster - 1),
                   label=f"Cluster {cluster}")
            for cluster in sorted(np.unique(labels))
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
        subtitle = "цвет — кластер k-means, k=3"
    ax.scatter(*(data[feature] for feature in features), c=colors,
               s=24 if name == "iris" else 17, alpha=0.75)
    ax.set(xlabel=features[0], ylabel=features[1], zlabel=features[2],
           title=f"{name}: {subtitle}")
    ax.view_init(elev=22, azim=-58)
    suffix = "original" if original else "clusters"
    fig.savefig(OUTPUT / f"task7_{name}_{suffix}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    iris = pd.read_csv(INPUT / "iris.csv", skiprows=1, header=None,
                       names=[*IRIS_FEATURES, "species"])
    baseball = pd.read_csv(INPUT / "baseball.csv")
    for name, data, features in (("iris", iris, IRIS_AXES),
                                 ("baseball", baseball, BASEBALL_AXES)):
        labels = load_labels(name, len(data))
        draw_3d(name, data, features, labels, original=True)
        draw_3d(name, data, features, labels, original=False)
        print(f"{name}: k=3, sizes={np.bincount(labels)[1:].tolist()}")
    print("Графики сохранены в", OUTPUT)


if __name__ == "__main__":
    main()

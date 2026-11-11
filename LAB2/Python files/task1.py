"""Пункт 1 ЛР № 2: исследование признаков и исходные 3D-диаграммы.

Запуск из корня LAB2: python "Python files/task1.py"
Результаты сохраняются в папке «Report files/task1».
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Входные"
OUTPUT = ROOT / "Report files" / "task1"
OUTPUT.mkdir(parents=True, exist_ok=True)

IRIS_FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
IRIS_AXES = [IRIS_FEATURES[0], IRIS_FEATURES[1], IRIS_FEATURES[3]]
BASEBALL_AXES = ["Height(inches)", "Weight(pounds)", "Age"]
POSITION_NAMES = {
    1: "Starting_Pitcher",
    2: "Catcher",
    3: "First_Baseman",
    4: "Outfielder",
    5: "Relief_Pitcher",
    6: "Second_Baseman",
    7: "Third_Baseman",
    8: "Shortstop",
    9: "Designated_Hitter",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Первая строка iris.csv хранит метаданные sklearn, затем идут 150 наблюдений.
    iris = pd.read_csv(
        INPUT / "iris.csv", skiprows=1, header=None,
        names=[*IRIS_FEATURES, "Вид"],
    )
    baseball = pd.read_csv(INPUT / "baseball.csv")
    required = {"Position", *BASEBALL_AXES}
    if not required.issubset(baseball.columns):
        raise ValueError(f"В baseball.csv отсутствуют столбцы: {required - set(baseball.columns)}")
    if iris[IRIS_FEATURES].isna().any().any() or baseball[BASEBALL_AXES].isna().any().any():
        raise ValueError("В числовых признаках есть пропуски: проверьте исходные данные")
    return iris, baseball


def save_correlation_map(iris: pd.DataFrame) -> pd.DataFrame:
    corr = iris[IRIS_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(4), IRIS_FEATURES, rotation=25, ha="right")
    ax.set_yticks(range(4), IRIS_FEATURES)
    for row in range(4):
        for col in range(4):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center")
    ax.set_title("Iris: корреляции числовых признаков (Пирсон)")
    fig.colorbar(image, ax=ax, label="Коэффициент корреляции")
    fig.savefig(OUTPUT / "task1_iris_correlations.png", dpi=180)
    plt.close(fig)
    return corr


def save_iris_3d(iris: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*(iris[feature] for feature in IRIS_AXES), c="#31688e", s=24, alpha=0.75)
    ax.set(xlabel=IRIS_AXES[0], ylabel=IRIS_AXES[1], zlabel=IRIS_AXES[2])
    ax.set_title("Iris: исходные наблюдения без видовых меток")
    ax.view_init(elev=22, azim=-58)
    fig.savefig(OUTPUT / "task1_iris_3d.png", dpi=180)
    plt.close(fig)


def save_baseball_3d(baseball: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    position = baseball["Position"].astype(int)
    palette = plt.get_cmap("tab10")
    colors = [palette((int(value) - 1) % 10) for value in position]
    ax.scatter(*(baseball[feature] for feature in BASEBALL_AXES), c=colors, s=18, alpha=0.7)
    ax.set(
        xlabel="Рост, дюймы", ylabel="Вес, фунты", zlabel="Возраст, годы",
        title="Бейсболисты: рост, вес, возраст; цвет — игровая позиция",
    )
    ax.view_init(elev=22, azim=-58)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=palette((code - 1) % 10),
               label=f"{code}: {POSITION_NAMES.get(code, 'Unknown_Position')}")
        for code in sorted(position.unique())
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.savefig(OUTPUT / "task1_baseball_3d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    iris, baseball = load_data()
    corr = save_correlation_map(iris)
    save_iris_3d(iris)
    save_baseball_3d(baseball)
    print(f"Iris: {len(iris)} наблюдений; бейсболисты: {len(baseball)} наблюдений")
    print("Корреляции Iris:\n", corr.round(3).to_string())
    print("Графики сохранены в:", OUTPUT)


if __name__ == "__main__":
    main()

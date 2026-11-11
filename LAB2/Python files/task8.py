"""Пункт 8 ЛР № 2: профили кластеров и итоговое сравнение.

Запуск из корня LAB2: python "Python files/task8.py"
Перед запуском требуется выполнить task6.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Входные"
REPORT = ROOT / "Report files"
OUTPUT = REPORT / "task8"
OUTPUT.mkdir(parents=True, exist_ok=True)
IRIS_FEATURES = ["sepal length (cm)", "sepal width (cm)",
                 "petal length (cm)", "petal width (cm)"]
BASEBALL_FEATURES = ["Height(inches)", "Weight(pounds)", "Age"]
POSITION_NAMES = {
    1: "Starting_Pitcher", 2: "Catcher", 3: "First_Baseman",
    4: "Outfielder", 5: "Relief_Pitcher", 6: "Second_Baseman",
    7: "Third_Baseman", 8: "Shortstop", 9: "Designated_Hitter",
}


def attach_labels(name: str, data: pd.DataFrame) -> pd.DataFrame:
    labels = pd.read_csv(REPORT / "task6" / f"task6_{name}_labels.csv")
    if len(labels) != len(data) or not np.array_equal(labels.row_id.to_numpy(), np.arange(len(data))):
        raise ValueError(f"Метки {name} не соответствуют исходным строкам")
    return data.assign(Cluster=labels.cluster.to_numpy(dtype=int))


def save_profile(name: str, data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    grouped = data.groupby("Cluster", sort=True)
    profile = grouped[features].mean()
    profile.insert(0, "size", grouped.size())
    profile.to_csv(OUTPUT / f"task8_{name}_profiles.csv", encoding="utf-8-sig", index_label="Cluster")
    standardized_means = (profile[features] - data[features].mean()) / data[features].std(ddof=0)

    if name == "iris":
        fig, (ax_heat, ax_extra) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True,
                                               gridspec_kw={"width_ratios": [2, 1]})
        ax_extra.bar([f"Cluster {i}" for i in profile.index], profile["size"], color="#6295ad")
        ax_extra.set(title="Размеры групп", ylabel="Число наблюдений")
        for i, size in enumerate(profile["size"]):
            ax_extra.text(i, size + 0.5, str(size), ha="center")
    else:
        fig, (ax_heat, ax_extra) = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True,
                                               gridspec_kw={"width_ratios": [1, 1.25]})
        position_names = data["Position"].astype(int).map(POSITION_NAMES)
        if position_names.isna().any():
            raise ValueError("Неизвестный код Position")
        counts = pd.crosstab(data["Cluster"], position_names)
        fractions = counts.div(counts.sum(axis=1), axis=0)
        counts.to_csv(OUTPUT / "task8_baseball_positions.csv", encoding="utf-8-sig", index_label="Cluster")
        fractions.plot(kind="bar", stacked=True, ax=ax_extra, cmap="tab10", width=0.75)
        ax_extra.set(title="Состав групп по Position", xlabel="Кластер", ylabel="Доля в группе", ylim=(0, 1))
        ax_extra.set_xticklabels([f"Cluster {i}" for i in profile.index], rotation=0)
        ax_extra.legend(title="Position", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    values = standardized_means.to_numpy()
    image = ax_heat.imshow(values, cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto")
    ax_heat.set(
        title="Средние признаки относительно всей выборки",
        xticks=np.arange(len(features)), xticklabels=features,
        yticks=np.arange(len(profile)), yticklabels=[f"Cluster {i}" for i in profile.index],
    )
    ax_heat.tick_params(axis="x", rotation=25)
    for row in range(len(profile)):
        for col in range(len(features)):
            ax_heat.text(col, row, f"{values[row, col]:+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax_heat, label="Отклонение среднего, стандартных отклонений")
    fig.suptitle(f"{name}: профили кластеров k-means при k=3", fontsize=15)
    fig.savefig(OUTPUT / f"task8_{name}_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return profile


def main() -> None:
    iris = pd.read_csv(INPUT / "iris.csv", skiprows=1, header=None,
                       names=[*IRIS_FEATURES, "species"])
    baseball = pd.read_csv(INPUT / "baseball.csv")
    for name, data, features in (("iris", iris, IRIS_FEATURES),
                                 ("baseball", baseball, BASEBALL_FEATURES)):
        profile = save_profile(name, attach_labels(name, data), features)
        print(f"{name}:\n{profile.round(2).to_string()}")
    metrics = pd.read_csv(REPORT / "task6" / "task6_results.csv")
    metrics[metrics.k.isin([2, 3, 4])].to_csv(OUTPUT / "task8_k_comparison.csv",
                                                 index=False, encoding="utf-8-sig")
    print("Результаты сохранены в", OUTPUT)


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = Path(__file__).resolve().parent.parent


history_size = 5
categorical_columns = [
    "Driver",
    "Compound",
    "Circuit"
]

numerical_columns = [f"LapTime_t-{i}" for i in range(1, history_size + 1)] + [
    "LapNumber",
    "Stint",
    "TyreLife",
    "AirTemp",
    "Humidity",
    "Rainfall",
    "TrackTemp",
    "Year"
]

@dataclass
class RaceConfig:
    circuit: str = "Austrian Grand Prix"
    driver: str = "VER"
    year: int = 2019
    total_laps: int = 58
    pit_loss: float = 22.0
    air_temp: float = 25.0
    humidity: float = 50.0
    rainfall: int = 0
    track_temp: float = 38.0
    default_baseline: float = 70.0

@dataclass
class Strategy:
    name: str
    start_compound: str
    pit_lap: Optional[int] = None
    new_compound: Optional[str] = None


def load_model():
    model = joblib.load(path / "models" / "xgboost_model.joblib")
    encoder = joblib.load(path / "models" / "onehot_encoder.joblib")
    return model, encoder


_baseline_cache = {}


def estimate_start_pace(cfg: RaceConfig, compound: str) -> float:
    # tempo startowe na nowych oponach (ponizej 5 okrazen na mieszance)
    key = (cfg.circuit, compound)
    if key in _baseline_cache:
        return _baseline_cache[key]

    values = []
    for f in (path / "data" / "final").glob(f"*/*_{cfg.circuit}.csv"):
        df = pd.read_csv(f, usecols=["LapTime", "Compound", "TyreLife"])
        df = df[(df["Compound"] == compound) & (df["TyreLife"] <= 5)]
        values.extend(df["LapTime"].dropna().tolist())

    result = float(np.median(values)) if values else cfg.default_baseline
    _baseline_cache[key] = result
    return result


def predict_lap(model, encoder, cfg, history, lap_number, stint, compound, tyre_life):
    # history - lista ostatnich czasow okrazen
    numerical = {f"LapTime_t-{i}": history[-i] for i in range(1, history_size + 1)}
    numerical.update(
        LapNumber=lap_number,
        Stint=stint,
        TyreLife=tyre_life,
        AirTemp=cfg.air_temp,
        Humidity=cfg.humidity,
        Rainfall=cfg.rainfall,
        TrackTemp=cfg.track_temp,
        Year=cfg.year,
    )
    x_num = np.array([[numerical[c] for c in numerical_columns]], dtype=float)

    cat_df = pd.DataFrame(
        [{"Driver": cfg.driver, "Compound": compound, "Circuit": cfg.circuit}]
    )[categorical_columns]
    x_cat = encoder.transform(cat_df)

    return float(model.predict(np.hstack([x_num, x_cat]))[0])

# Symulacja wyscigu dla danej strategii
def simulate_race(model, encoder, cfg: RaceConfig, strategy: Strategy):
    compound = strategy.start_compound
    stint = 1
    tyre_life = 1
    history = [estimate_start_pace(cfg, compound)] * history_size

    lap_times = []
    compounds = []

    for lap in range(1, cfg.total_laps + 1):
        pred = predict_lap(model, encoder, cfg, history, lap, stint, compound, tyre_life)
        history.append(pred)
        compounds.append(compound)

        lap_total = pred
        if strategy.pit_lap is not None and lap == strategy.pit_lap:
            lap_total += cfg.pit_loss
            compound = strategy.new_compound
            stint += 1
            tyre_life = 1
            # nowy stint nie ma historii, ustalam tempo startowe na nowej mieszanki
            history = [estimate_start_pace(cfg, compound)] * history_size
        else:
            tyre_life += 1

        lap_times.append(lap_total)

    return {
        "name": strategy.name,
        "lap_times": np.array(lap_times),
        "cumulative": np.cumsum(lap_times),
        "compounds": compounds,
        "total_time": float(np.sum(lap_times)),
    }

# porownanie strategii
def compare_strategies(model, encoder, cfg, strategies):
    results = [simulate_race(model, encoder, cfg, s) for s in strategies]
    best = min(results, key=lambda r: r["total_time"])

    print(f"\nWyścig: {cfg.circuit}, {cfg.driver}, {cfg.year}, {cfg.total_laps} okrążeń, "
          f"strata pit-stopu {cfg.pit_loss:.1f} s")
    print(f"{'Strategia':<22}{'Czas całkowity [s]':>20}{'Różnica do najlepszej [s]':>28}")
    for r in sorted(results, key=lambda r: r["total_time"]):
        print(f"{r['name']:<22}{r['total_time']:>20.2f}{r['total_time'] - best['total_time']:>28.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # całkowity czas wyścigu (jako różnica do najlepszej strategii)
    names = [r["name"] for r in results]
    deltas = [r["total_time"] - best["total_time"] for r in results]
    colors = ["tab:green" if r is best else "tab:blue" for r in results]
    bars = axes[0].bar(names, deltas, color=colors)
    for bar, r in zip(bars, results):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{r['total_time']:.1f} s", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Strata do najlepszej strategii [s]")
    axes[0].set_title("Całkowity czas wyścigu (etykiety = czas absolutny)")
    axes[0].tick_params(axis="x", rotation=25)

    # przebieg różnicy skumulowanego czasu względem najlepszej strategii
    laps = np.arange(1, cfg.total_laps + 1)
    for r, s in zip(results, strategies):
        axes[1].plot(laps, r["cumulative"] - best["cumulative"], label=r["name"])
        if s.pit_lap:
            axes[1].axvline(s.pit_lap, color="grey", alpha=0.15, linestyle=":")
    axes[1].set_xlabel("Okrążenie")
    axes[1].set_ylabel("Różnica czasu skumulowanego do najlepszej [s]")
    axes[1].set_title("Przebieg wyścigu względem najlepszej strategii")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = path / "results"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "strategy_comparison.png", dpi=150)
    return results


def sweep_pit_lap(model, encoder, cfg, start, new, lap_range=range(8, 25)):
    # szukamy optymalnego okrazenia zjazdu
    laps, totals = [], []
    for pit_lap in lap_range:
        s = Strategy(f"pit {pit_lap}", start, pit_lap, new)
        totals.append(simulate_race(model, encoder, cfg, s)["total_time"])
        laps.append(pit_lap)

    best_idx = int(np.argmin(totals))
    plt.figure(figsize=(8, 4.5))
    plt.plot(laps, totals, marker="o", markersize=3)
    plt.scatter([laps[best_idx]], [totals[best_idx]], color="tab:green", zorder=3,
                label=f"optimum: okrążenie {laps[best_idx]}")
    plt.xlabel("Okrążenie pit-stopu")
    plt.ylabel("Całkowity czas wyścigu [s]")
    plt.title(f"{start} → {new}: wpływ momentu zjazdu")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = path / "results"
    out.mkdir(exist_ok=True)
    plt.savefig(out / "pit_lap_sweep.png", dpi=150)
    print(f"\nNajlepsze okrążenie zjazdu ({start}→{new}): {laps[best_idx]} "
          f"({totals[best_idx]:.2f} s)")


if __name__ == "__main__":
    cfg = RaceConfig()
    model, encoder = load_model()

    strategies = [
        Strategy("S→H, pit 10", "SOFT", 10, "HARD"),
        Strategy("S→H, pit 15", "SOFT", 15, "HARD"),
        Strategy("S→H, pit 18", "SOFT", 18, "HARD"),
        Strategy("S→H, pit 21", "SOFT", 21, "HARD"),
        Strategy("S→H, pit 23", "SOFT", 23, "HARD"),
    ]

    compare_strategies(model, encoder, cfg, strategies)
    sweep_pit_lap(model, encoder, cfg, "SOFT", "HARD")
    plt.show()
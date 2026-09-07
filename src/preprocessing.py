import pandas as pd
from pathlib import Path

# Pobieranie danych z wyścigów z konkretnych lat
years = [2019,2020,2021,2022,2023,2024,2025]

path = Path(__file__).resolve().parent.parent

for year in years:
    # Sciezki wejsciowe i wyjsciowe folderow
    input_dir = path / "data" / "proceed" / str(year)
    output_dir = path / "data" / "final" / str(year)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pobranie wszystkich pliki CSV z danego roku
    for file in input_dir.glob("*.csv"):
        if file.name.endswith("_weather.csv"):
            continue

        try:
            # Wczytanie danych dotyczacych okrazen
            laps = pd.read_csv(file)

            weather_file = file.with_name(
                file.stem + "_weather.csv"
            )

            if not weather_file.exists():
                print(f"Brak danych pogodowych: {weather_file.name}")
                continue

            weather = pd.read_csv(weather_file)

            # Zwraca roznice czasu
            laps["Time"] = pd.to_timedelta(
                laps["Time"]
            )

            weather["Time"] = pd.to_timedelta(
                weather["Time"]
            )

            laps = laps.sort_values("Time")
            weather = weather.sort_values("Time")

            # Polaczenie dwoch plikow (okrazenia w czasie wyscigu i pogoda po czasie sesji)
            merged = pd.merge_asof(
                laps,
                weather,
                on="Time",
                direction="backward"
            )

            merged["Rainfall"] = merged["Rainfall"].astype(int)
            merged["LapTime"] = pd.to_timedelta(merged["LapTime"]).dt.total_seconds()
            merged = merged.drop(columns="Time")

            output_file = output_dir / file.name

            merged.to_csv(
                output_file,
                index=False
            )

        except Exception as e:

            print(f"Błąd w pliku {file.name}: {e}")

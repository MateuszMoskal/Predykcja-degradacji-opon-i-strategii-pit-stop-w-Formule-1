import pandas as pd
from pathlib import Path

# Pobieranie danych z wyścigów z konkretnych lat
years = [2019,2020,2021,2022,2023,2024,2025]

# Dane wybrane do procesu uczenia dotyczące okrążeń
lap_columns = [
    "Time",
    "Driver",
    "LapTime",
    "LapNumber",
    "Stint",
    "Compound",
    "TyreLife"
]

# Dane wybrane do procesu uczenia dotyczące pogody
weather_columns = [
    "Time",
    "AirTemp",
    "Humidity",
    "Rainfall",
    "TrackTemp"
]

path = Path(__file__).resolve().parent.parent


for year in years:
    # Sciezki wejsciowe i wyjsciowe folderow
    input_dir = path / "data" / "raw" / str(year)
    output_dir = path / "data" / "proceed" / str(year)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pobranie wszystkich pliki CSV z danego roku
    for file in input_dir.glob("*.csv"):
        if file.name.endswith("_weather.csv"):
            continue

        try:
            # Wczytanie danych dotyczacych okrazen
            laps = pd.read_csv(file)

            # Usuniecie okrazen wyjazdowych
            laps = laps[laps["PitOutTime"].isna()]

            # Usuniecie okrazen zjazdowych
            laps = laps[laps["PitInTime"].isna()]

            # Usunięcie okrazen pod samochodem bezpieczenstwa lub czerwona flaga
            laps = laps[
                ~laps["TrackStatus"]
                .astype(str)
                .str.contains("4|5", na=False)
            ]

            laps = laps[lap_columns]

            #Zapis do pliku (proceed/year)
            output_file = output_dir / file.name

            laps.to_csv(
                output_file,
                index=False
            )

            # Wczytanie danych pogodowych
            weather_input_file = file.with_name(
                file.stem + "_weather.csv"
            )
            if weather_input_file.exists():

                weather = pd.read_csv(weather_input_file)
                weather = weather[
                    weather_columns
                ].copy()
                # Zapisanie danych pogodowych do pliku proceed/year
                weather_output_file = (
                    output_dir /
                    weather_input_file.name
                )

                weather.to_csv(
                    weather_output_file,
                    index=False
                )

            else:
                print(
                    f"Nie znaleziono pliku pogodowego: "
                    f"{weather_input_file.name}"
                )

        except Exception as e:

            print(f"Błąd w pliku {file.name}: {e}")
import fastf1
from pathlib import Path

years = [2021, 2022, 2023, 2024, 2025]
for year in years:
    schedule = fastf1.get_event_schedule(year)
    print(schedule)
    for _, event in schedule.iterrows():
        number = event["RoundNumber"]
        gp = event["EventName"]
        path = Path(__file__).resolve().parent.parent
        output = path / "data" / str(year)
        output.mkdir(parents=True, exist_ok=True)
        output_file = output/f"{number}_{gp}.csv"
        weather_file = output/f"{number}_{gp}_weather.csv"
        try:
            session = fastf1.get_session(year, number, "R")
            session.load()
            laps = session.laps
            weather = session.weather_data
            laps.to_csv(output_file, index=False)
            weather.to_csv(weather_file, index=False)
        except Exception as e:
            print(f"{e}")
            print(f"{gp}")
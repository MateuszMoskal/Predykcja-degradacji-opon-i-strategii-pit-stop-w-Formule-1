import fastf1
import os

# Create cache directory if it doesn't exist
cache_dir = os.path.expanduser('~/fastf1_cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Enable the cache
fastf1.Cache.enable_cache(cache_dir)

# Now you can load sessions
session = fastf1.get_session(2024, 'Austria', 'R')

# Load all session data (timing, telemetry, weather, etc.)
session.load()

# Get all laps for a specific driver
alonso_laps = session.laps.pick_drivers('ALO')

# Filter for quick laps only (removes outliers)
quick_laps = alonso_laps.pick_quicklaps()

# Get the fastest lap
fastest_lap = alonso_laps.pick_fastest()

# print(f"Fastest lap time: {fastest_lap['LapTime']}")
# print(f"Lap number: {fastest_lap['LapNumber']}")
# print(f"Compound: {fastest_lap['Compound']}")
# print(f"columns_alonso: {alonso_laps.columns}")
print(f"alonso_laps:  {alonso_laps.LapNumber}, {alonso_laps.LapTime},{alonso_laps.Compound}, {alonso_laps.TyreLife}, {alonso_laps.FreshTyre}, {alonso_laps.TrackStatus.to_string()}")
print(f"columns_alonso: {alonso_laps}", f"quick_laps : {quick_laps}") #quick_laps usuwa dane podczas safety_cara - sprawdzić czy usuwa dane z okrążeń wyjazdowych i zjazdowych.

# driver_laps = session.laps.pick_drivers('ALO').pick_quicklaps()
#
# # Show compound usage
# compound_usage = driver_laps.groupby('Compound').size()
# print(f"\nCompound usage:\n{compound_usage}")
#
# medium_laps = session.laps[session.laps['Compound'] == 'MEDIUM']
# print(f"Compound Time: {medium_laps}")
#
# weather = session.weather_data
# print(weather[['Time', 'AirTemp', 'TrackTemp', 'Rainfall']].head())

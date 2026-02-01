import pandas as pd
import fastf1 as f

def get_calendar():
    races = {}
    years = [2018, 2019, 2021, 2022, 2023, 2024, 2025]

    for year in range(0, len(years)):
        get_schedule = f.get_event_schedule(years[year])

        races[years[year]] = get_schedule['Country']
    
    return races


def get_race_data(year, track):

    rows = []

    try:
        race = f.get_session(year, track, 'R')
        race.load()  # loads all lap and telemetry data
    except:
        print("Error loading track")
        return []
    total_laps = race.total_laps # Returns total laps of the race

    laps = race.laps # Data for all the laps
    total_temp_rows = len(race.weather_data['AirTemp']) # Rows of weather data
    # Increment value for weather data
    temp_rows = float(total_temp_rows) / float(total_laps)
    # Dictionary to store gaps per lap
    leader_gaps = {}

    weather_count = temp_rows
    race_winner = race.results.iloc[0]['DriverNumber']

    for lap_number in range(1, total_laps + 1):

        #Get all driver data for the current lap
        lap_driver_data = race.laps[laps['LapNumber'] == lap_number]

        #Get the person in first place for the current lap
        first_place = lap_driver_data[lap_driver_data['Position'] == 1]
        second_place = lap_driver_data[lap_driver_data['Position'] == 2]

        if first_place.empty:
            continue

    
        laps_remaining = total_laps - lap_number # Get remaining laps
        # Get gap to second place for current lap
        gap_to_p2 = (second_place.Time.values[0] - first_place.Time.values[0]) / pd.Timedelta(seconds=1)
        leader_gaps[lap_number] = gap_to_p2

        # Logic to work out gap trend over three laps
        if lap_number > 3:
            gap_trend_3 = (gap_to_p2 - leader_gaps[lap_number - 3]) / 3
        else:
            gap_trend_3 = 0


        tire_compound = first_place.Compound.values[0] # Current lap tire compound
        tire_age = first_place.TyreLife.values[0] # Current lap tire age
        status = first_place['TrackStatus'].values[0]
        safety_car = 1 if status in ['SC', 'VSC'] else 0 #Current lap safety car

        if weather_count >= total_temp_rows - 1:
            weather_count = total_temp_rows - 1

        # Current lap air temp
        air_temp = race.weather_data.iloc[round(weather_count), 1]
        # Current lap track temp
        track_temp = race.weather_data.iloc[round(weather_count), 5]
        # Current lap rain check (is it raining this lap)
        rain_raw = race.weather_data.iloc[round(weather_count), 4]
        rain = 1 if rain_raw else 0
        weather_count += temp_rows

        label = 1 if first_place['DriverNumber'].values[0] == race_winner else 0

        rows.append({
            'track': track,
            'lap': lap_number,
            'laps_remaining': laps_remaining,
            'gap_to_p2': gap_to_p2,
            'gap_trend_3': gap_trend_3,
            'tire_compound': tire_compound,
            'tire_age': tire_age,
            'safety_car': safety_car,
            'air_temp': air_temp,
            'track_temp': track_temp,
            'rain': rain,
            'label': label
        })

    return rows


#table = []
#races = get_calendar()

#for year, tracks in races.items():
    #for track in tracks:
        #print(f'[FETCHING DAT FOR {year} and {track}]')
        #table.extend(get_race_data(year, track))

#df = pd.DataFrame(table)
#df.to_csv('f1_data.csv', index=True)
#print(df)
print(get_race_data(2025, "Silverstone"))







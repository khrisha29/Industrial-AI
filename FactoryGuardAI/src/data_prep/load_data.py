import pandas as pd

def load_sensor_data(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp')
    return df

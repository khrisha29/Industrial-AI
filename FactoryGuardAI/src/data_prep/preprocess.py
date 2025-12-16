import pandas as pd

def load_raw_data(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def create_temporal_features(df):
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    return df

def create_rolling_features(df):
    df = df.sort_values('timestamp')

    df['temp_mean_5'] = df['temperature'].rolling(window=5).mean()
    df['vib_std_10'] = df['vibration'].rolling(window=10).std()
    df['pressure_max_30'] = df['pressure'].rolling(window=30).max()

    return df

def create_future_target(df):
    df = df.sort_values('timestamp')
    df['failure_future'] = df['failure'].shift(-1440)  # 24 hours = 1440 min if 1-min data
    return df

def preprocess_pipeline(path):
    df = load_raw_data(path)
    df = create_temporal_features(df)
    df = create_rolling_features(df)
    df = create_future_target(df)
    df = df.dropna()  # Required for ML
    return df

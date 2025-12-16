def create_rolling_features(df):
    """
    Create rolling window and exponential moving average features
    for sensor data.
    """
    windows = [2,3]

    for window in windows:
        # Rolling mean
        df[f"temp_roll_mean_{window}"] = df["temperature"].rolling(window).mean()
        df[f"vib_roll_mean_{window}"] = df["vibration"].rolling(window).mean()
        df[f"press_roll_mean_{window}"] = df["pressure"].rolling(window).mean()

        # Rolling standard deviation
        df[f"temp_roll_std_{window}"] = df["temperature"].rolling(window).std()
        df[f"vib_roll_std_{window}"] = df["vibration"].rolling(window).std()
        df[f"press_roll_std_{window}"] = df["pressure"].rolling(window).std()

        # Exponential Moving Average
        df[f"temp_ema_{window}"] = df["temperature"].ewm(span=window, adjust=False).mean()
        df[f"vib_ema_{window}"] = df["vibration"].ewm(span=window, adjust=False).mean()
        df[f"press_ema_{window}"] = df["pressure"].ewm(span=window, adjust=False).mean()

    return df

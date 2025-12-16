def create_lag_features(df, cols, lags=[1, 2]):
    for col in cols:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df

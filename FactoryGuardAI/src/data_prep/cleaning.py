def clean_data(df):
    df = df.set_index('timestamp')
    df = df.interpolate(method='time')
    df = df.fillna(method='bfill')
    df = df.fillna(method='ffill')
    return df.reset_index()

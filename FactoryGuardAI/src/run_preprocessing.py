from data_prep.preprocess import preprocess_pipeline

if __name__ == "__main__":
    input_path = "data/raw/sensor_data.csv"
    output_path = "data/processed/processed_data.csv"

    df = preprocess_pipeline(input_path)
    df.to_csv(output_path, index=False)

    print("Preprocessing completed successfully")
    print("Processed data saved to:", output_path)

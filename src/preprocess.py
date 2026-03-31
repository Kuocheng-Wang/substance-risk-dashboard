import pandas as pd
from pathlib import Path


def preprocess():
    project_root = Path(__file__).resolve().parent.parent
    input_file = project_root / "data" / "raw" / "drugsComTrain_raw.csv"
    output_dir = project_root / "data" / "processed"
    output_file = output_dir / "cleaned_data.csv"

    print("Reading file:", input_file)

    df = pd.read_csv(input_file)

    # Keep only the required columns
    df = df[["uniqueID", "date", "review"]].copy()

    # Rename columns
    df.rename(columns={
        "uniqueID": "post_id",
        "review": "text"
    }, inplace=True)

    # Drop missing values
    df = df.dropna(subset=["date", "text"])

    # Process date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Clean text
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    # Remove duplicate text rows
    df = df.drop_duplicates(subset=["text"])

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export cleaned data
    df.to_csv(output_file, index=False, encoding="utf-8")

    print("cleaned_data.csv has been generated successfully!")
    print(df.head())
    print("\nTotal rows:", len(df))
    print("\nOutput file:", output_file)


if __name__ == "__main__":
    preprocess()

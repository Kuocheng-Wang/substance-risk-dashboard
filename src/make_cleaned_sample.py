import pandas as pd
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent
input_file = base_path / "data" / "processed" / "cleaned_data.csv"
output_file = base_path / "data" / "processed" / "cleaned_data_sample.csv"

df = pd.read_csv(input_file)
df.head(1000).to_csv(output_file, index=False)

print("saved cleaned_data_sample.csv")
print(output_file)

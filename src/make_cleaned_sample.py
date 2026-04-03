import pandas as pd
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent
input_file = base_path / "data" / "processed" / "predictions_drugsComTrain.csv"
output_file = base_path / "data" / "processed" / "predictions_drugsComTrain_sample.csv"

df = pd.read_csv(input_file)
df.head(1000).to_csv(output_file, index=False)

print("saved cleaned_data_sample.csv")
print(output_file)

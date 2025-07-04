import pandas as pd
import os

# Set input file path (.sav) and output directory
sav_file_path = r"C:\BULabAssets\BULabProjects\ColloborationWithDrYoon\dataset\Homeless data.sav"             # ← Replace with actual .sav file path
output_dir = r"C:\BULabAssets\BULabProjects\ColloborationWithDrYoon\dataset"                    # ← Replace with your target folder
csv_file_name = 'homeless_data.csv'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the .sav file
#df, meta = pd.read_spss(sav_file_path)
# Load only the DataFrame
df = pd.read_spss(sav_file_path)

# Save as CSV
csv_path = os.path.join(output_dir, csv_file_name)
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"CSV file saved to: {csv_path}")

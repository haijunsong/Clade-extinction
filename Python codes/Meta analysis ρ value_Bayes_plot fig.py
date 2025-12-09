import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os

# Input data file path
# file_path = r'F:\1-投稿\2022-clades灭绝\Pyrate\12 clades 2025-11\output1myr\Meta_Analysis_Results_Pyrate.xlsx'
file_path = 'F:/1-投稿/2022-clades灭绝/PBDB Data/SQS results/Meta_Analysis_Results_SQS.xlsx'
# Read data
df = pd.read_excel(file_path)

# Extract relevant columns
variables = df['Variable'].astype(str)
y = df['Combined_rho_Bayesian']
yerr_lower = y - df['CI_Lower_rho_Bayesian']
yerr_upper = df['CI_Upper_rho_Bayesian'] - y
yerr = [yerr_lower, yerr_upper]
pvals = df['p_value_Bayesian']

# Create the figure
plt.figure(figsize=(10, 6))
plt.errorbar(variables, y, yerr=yerr, fmt='o', capsize=5, elinewidth=2, color='b')

# Mark points with p_value_Bayesian < 0.01 with a red star
for i, (var, val, p) in enumerate(zip(variables, y, pvals)):
    if p < 0.01:
        plt.text(i, val + (yerr_upper.iloc[i] if hasattr(yerr_upper, 'iloc') else yerr_upper[i]) + 0.03,
                 '*', ha='center', va='bottom', color='red', fontsize=18, fontweight='bold')

plt.xticks(rotation=45)
plt.xlabel('Variable')
plt.ylabel('Combined_rho_Bayesian')
plt.title('Combined_rho_Bayesian with 95% Confidence Interval')
plt.ylim(-0.8, 1)  # Set y-axis minimum to -0.6 and maximum to 1

plt.tight_layout()

# Automatically generate output path and name
output_dir = os.path.dirname(file_path)
basename = os.path.splitext(os.path.basename(file_path))[0]
output_file = os.path.join(output_dir, basename + '.pdf')
plt.savefig(output_file, dpi=300)

print(f"Figure saved as {output_file}")

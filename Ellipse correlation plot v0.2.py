import matplotlib
matplotlib.use('TkAgg')  # Ensure using an interactive backend

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# Read different sheets from the Excel file and set the first column as index
file_path = 'F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-3/output1myr/Spearman correlation_results pyrate v0.2.xlsx'
r_values = pd.read_excel(file_path, sheet_name='Correlation Coefficients', index_col=0)
p_values = pd.read_excel(file_path, sheet_name='P Values', index_col=0)

# Print DataFrame shape and column names for debugging
print("r_values DataFrame shape:", r_values.shape)
print("r_values DataFrame columns:", r_values.columns)
print("p_values DataFrame shape:", p_values.shape)
print("p_values DataFrame columns:", p_values.columns)

# Ensure r_values and p_values have the same shape
if r_values.shape != p_values.shape:
    raise ValueError("r_values and p_values must have the same shape")

# Convert r values and p values to float
r_values = r_values.apply(pd.to_numeric, errors='coerce')
p_values = p_values.apply(pd.to_numeric, errors='coerce')

# Create ellipse correlation plot
fig, ax = plt.subplots(figsize=(10, 10))

# Define colormap
cmap = plt.get_cmap('coolwarm')

# Iterate over each biological category and environmental factor
for i in range(r_values.shape[0]):
    for j in range(r_values.shape[1]):
        r = r_values.iloc[i, j]
        p = p_values.iloc[i, j]

        # Check for missing values
        if pd.isna(r) or pd.isna(p):
            continue

        # Ellipse parameters
        width = np.sqrt(1 - abs(r))
        height = 1.0
        angle = 45 if r > 0 else -45

        # Color
        color = cmap((r + 1) / 2)

        # Draw ellipse
        ellipse = Ellipse((j, i), width, height, angle=angle, color=color, alpha=0.7)
        ax.add_patch(ellipse)

        # Add r value and p value label with '*' and '**' based on p-value
        label = f"{r:.2f}"
        if p < 0.004:
            label += "*"

        # No addition if p >= 0.05

        ax.text(j, i, label, ha='center', va='center', fontsize=8)

# Set tick labels
ax.set_xticks(np.arange(r_values.shape[1]))
ax.set_yticks(np.arange(r_values.shape[0]))
ax.set_xticklabels(r_values.columns, rotation=90)  # Rotate x-axis labels
ax.set_yticklabels(r_values.index)

# Set axis limits
ax.set_xlim(-0.5, r_values.shape[1] - 0.5)
ax.set_ylim(-0.5, r_values.shape[0] - 0.5)
ax.invert_yaxis()
ax.set_aspect('equal')

# Show color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Correlation Coefficient (ρ)')

plt.title('')
plt.tight_layout()

# Save the figure as an SVG with text converted to paths
plt.savefig("ellipse_correlation_plot.svg", format='svg', bbox_inches='tight', pad_inches=0.1)
print("Figure saved to 'ellipse_correlation_plot.svg'")

# Display the plot
plt.show()
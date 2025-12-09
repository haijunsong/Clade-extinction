import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ensure a compatible backend
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def calculate_correlation(input_file):
    try:
        # Read the Excel file
        data = pd.read_excel(input_file)
        print("Data loaded successfully. Preview:")
        print(data.head())

        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[float, int])
        print("\nPreview of numeric data:")
        print(numeric_data.head())

        columns = numeric_data.columns
        num_columns = len(columns)

        # Create matrices for correlation, p-value, and sample size
        correlation_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)
        p_value_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)
        n_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)  # Sample size

        for i in range(num_columns):
            for j in range(i + 1, num_columns):
                col1 = columns[i]
                col2 = columns[j]
                valid_data = numeric_data[[col1, col2]].dropna()
                N = len(valid_data)
                if N > 2:  # At least 3 data points required
                    r, p = pearsonr(valid_data[col1], valid_data[col2])
                    correlation_matrix.loc[col1, col2] = r
                    correlation_matrix.loc[col2, col1] = r
                    p_value_matrix.loc[col1, col2] = p
                    p_value_matrix.loc[col2, col1] = p
                    n_matrix.loc[col1, col2] = N
                    n_matrix.loc[col2, col1] = N
                else:
                    print(f"Warning: Not enough valid data points between {col1} and {col2} to calculate correlation.")

            # Fill diagonal values
            correlation_matrix.iloc[i, i] = 1
            p_value_matrix.iloc[i, i] = 0
            n_matrix.iloc[i, i] = numeric_data[columns[i]].count()

        print("\nCorrelation matrix:")
        print(correlation_matrix)
        print("\nP-value matrix:")
        print(p_value_matrix)
        print("\nN (sample size) matrix:")
        print(n_matrix)

        # Save the results to an Excel file
        # with pd.ExcelWriter("Correlation_results_origin_SQS.xlsx") as writer:
        with pd.ExcelWriter("Correlation_results_origin_PyRate.xlsx") as writer:
            correlation_matrix.to_excel(writer, sheet_name="Correlation Coefficients")
            p_value_matrix.to_excel(writer, sheet_name="P Values")
            n_matrix.to_excel(writer, sheet_name="Number")

        # Visualization
        visualize_correlation(correlation_matrix, p_value_matrix, n_matrix)

    except Exception as e:
        print(f"An error occurred: {e}")

def visualize_correlation(correlation_matrix, p_value_matrix, n_matrix):
    columns = correlation_matrix.columns
    num_columns = len(columns)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-0.5, num_columns - 0.5)
    ax.set_ylim(-0.5, num_columns - 0.5)

    for i in range(num_columns):
        for j in range(i + 1, num_columns):
            r = correlation_matrix.iloc[i, j]
            p = p_value_matrix.iloc[i, j]
            n = n_matrix.iloc[i, j]

            width = 1.0
            height = abs(r) * width if pd.notnull(r) else 0

            color = 'blue' if pd.notnull(p) and p <= 0.05 else 'none'

            ellipse = Ellipse((i, j), width, height, facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(ellipse)

            # Show r, p, n
            ax.text(i, j, f"{r:.2f}\np={p:.3f}\nN={int(n) if pd.notnull(n) else 'NA'}",
                    ha='center', va='center', fontsize=8)

    ax.set_xticks(range(num_columns))
    ax.set_yticks(range(num_columns))
    ax.set_xticklabels(columns, rotation=90)
    ax.set_yticklabels(columns)
    ax.set_title("Correlation, Significance & Sample Size Visualization", fontsize=14)
    plt.tight_layout()
    plt.savefig("correlation_visualization.png")
    print("Figure saved to 'correlation_visualization.png'")

# Example usage
if __name__ == "__main__":
    # input_file = r"F:\1-投稿\2022-clades灭绝\PBDB Data\SQS results\SQS result vs Environmental factros v0.5.xlsx"
    input_file = r"F:\1-投稿\2022-clades灭绝\Pyrate\12 clades 2025-11\output1myr\Pyrate result vs environmental factors v0.3.xlsx"
    calculate_correlation(input_file)



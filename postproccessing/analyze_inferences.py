import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel file into a DataFrame
file_path = 'ablation_results.xlsx'  # Replace with your Excel file path
df = pd.read_excel(file_path)

# Extract the relevant columns
x_col = 'Landing Area % (d=n/a)'  # x-axis column
y_col_d4 = 'Landing Area % (d=4)'  # y-axis column for d=4
y_col_d15 = 'Landing Area % (d=15)'  # y-axis column for d=15

# Check if required columns exist
if x_col not in df.columns or y_col_d4 not in df.columns or y_col_d15 not in df.columns:
    raise ValueError(f"One or more required columns are missing in the DataFrame.")

# Plotting
plt.figure(figsize=(12, 8))

# Scatter plot and line of best fit for d=4
plt.scatter(df[x_col], df[y_col_d4], color='green', label='Landing Area % (d=4)', marker='o')
coeffs_d4 = np.polyfit(df[x_col], df[y_col_d4], 1)  # Linear fit (degree 1)
poly_d4 = np.poly1d(coeffs_d4)
plt.plot(df[x_col], poly_d4(df[x_col]), color='green', linestyle='--')

# Scatter plot and line of best fit for d=15
plt.scatter(df[x_col], df[y_col_d15], color='red', label='Landing Area % (d=15)', marker='x')
coeffs_d15 = np.polyfit(df[x_col], df[y_col_d15], 1)  # Linear fit (degree 1)
poly_d15 = np.poly1d(coeffs_d15)
plt.plot(df[x_col], poly_d15(df[x_col]), color='red', linestyle='--')

# Customize the plot
plt.xlabel(x_col)
plt.ylabel('Landing Area %')
plt.title('Landing Area % vs Landing Area % (d=4 and d=15)')
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig("landing_area_comparison_with_fit.png")
plt.show()
